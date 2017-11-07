import networkx as nx
#from graph_tool.all import graph_tool as gt
import pickle
from shapely.prepared import prep
from shapely.ops import snap, linemerge, nearest_points
from shapely.geometry import MultiLineString, LineString, Point, Polygon, GeometryCollection
# from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import chain
from glob import glob
import sys
import timeit
import os
from io import StringIO

import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import random
import matplotlib.pyplot as plt
import matplotlib as mpl


# 1. road
road = gpd.read_file('../data/nWMDmap2/rdsclip1.shp')

# 2. demographics: 
dp = gpd.read_file('../data/nWMDmap2/censusclip1.shp').set_index('GEOID10')
#the cleaned shape data might be a clipped area(which means that some tracts selected were cut by our artificial border)
#so for these tracts on border, we calculate the proportion to be counted inside of "our area".
dp['portion'] = dp.apply(lambda tract: tract.geometry.area / tract.Shape_Area, axis=1)

# 3. commute:
od = pd.read_csv('../data/tract-od.csv.zip',dtype={i:(str if i<2 else int) for i in range(6)})

# 4. number of workplaces:
cbp = pd.read_csv('../data/cbp10co.zip')
cbp = cbp[(cbp.naics.str.startswith('-'))] #All types of establishments included
cbp['fips'] = cbp.fipstate.map("{:02}".format) + cbp.fipscty.map("{:03}".format)
cbp = cbp.set_index('fips')

# 5. schools + daycares:
school = gpd.read_file('../data/education/education.shp')


# add two columns on dp (demographics) dataframe
# each tract: [col1]how many workplaces, and [col2] the probability for employees to be in each workplace

def number_of_wp(dp, od, cbp):
    """
    calculate number of workplaces for each tract
    wp_tract = wp_cty * (tract_employed / county_employed)
    """
    # get the number of employees per tract
    dwp = od[['work','S000']].groupby('work').sum()
    dwp = pd.merge(dp.portion.to_frame(),dwp,left_index=True,right_index=True,how='left').fillna(0)
#     dwp = dwp.portion*dwp.S000/10
    wp_class = ["n1_4","n5_9","n10_19","n20_49","n50_99","n100_249","n250_499","n500_999","n1000","n1000_1","n1000_2","n1000_3","n1000_4"]
    dwp['county'] = dwp.index.str[:5]
    a = dwp.groupby('county').sum()
    a = a.join(cbp[wp_class].sum(axis=1).to_frame('wpcty'))
    # note: as Dr. Crooks suggested agents not living in our region included
    dwp = (dwp.portion * dwp.S000 / dwp.county.map(a.S000)) * dwp.county.map(a.wpcty)
    return dwp.apply(np.ceil).astype(int)

def wp_proba(x):
    """
    probability of an employee working in that workplace is lognormal:
    http://www.haas.berkeley.edu/faculty/pdf/wallace_dynamics.pdf
    """
    if x == 0: return np.zeros(0)
    b = np.random.lognormal(mean=2,size=x).reshape(-1, 1)
    return np.sort(normalize(b,norm='l1',axis=0).ravel())
    
# number of workplaces in each track
dp['WP_CNT'] = number_of_wp(dp,od,cbp)
# each track, the probability distribution for an employee to work there is lognormal
# this column has the probability for each workplace and they add up to 1. 
dp['WP_PROBA'] = dp.WP_CNT.map(wp_proba)

dp.sort_index(axis=1,inplace=True)
dp['portion']=round(dp['portion'],2)

# 1. create individuals
def create_individuals(tract):
    """Generate a population of ages and sexes as a DataFrame

    Given the number of individuals for 18 age groups of each sex,
    return a two column DataFrame of age ([0,89]) and sex ('m','f')
    """
    portion = tract.geometry.area / tract.Shape_Area # what portion of the tract is included
    portion = round(portion,2)
    age_sex_groups = (tract[22:59].drop('DP0010039') * portion).astype(int)
    dfs=[]
    for code,count in enumerate(age_sex_groups):
        base_age = (code % 18)*5
        gender = 'm' if code < 18 else 'f'
        ages = []
        for offset in range(4):
            ages.extend([offset+base_age]*(count//5))
        ages.extend([base_age+4]*(count-len(ages)))
        dfs.append(pd.DataFrame({'code':code, 'age':ages,'sex':[gender]*count}))
    df = pd.concat(dfs).sample(frac=1,random_state=123).reset_index(drop=True)
    df.index = tract.name + 'i' + df.index.to_series().astype(str)
    df['friends'] = [set()] * len(df)
    return df

#2. create households
def create_households(tract, people):
    hh_cnt = get_hh_cnts(tract)
    hholds = pd.DataFrame()
    #create empty households with its types
    hholds['htype'] = np.repeat(hh_cnt.index,hh_cnt)
    hholds = hholds[hholds.htype != 6].sort_values('htype',ascending=False).append(hholds[hholds.htype == 6])
    #create member for each households; 
    #for remaining populating, populate them in households as relatives and those living in group quarter (non-household)
    populate_households(tract, people, hholds)


def get_hh_cnts(tract):
    """
    Eleven household types:
    0         h&w (no<18)
    1      h&w&ch (ch<18)
    2        male (no<18)
    3        male (ch<18)
    4      female (no<18)
    5      female (ch<18)
    6     nonfamily group
    7       lone male <65
    8       lone male >65
    9      lone female<65
    10     lone female>65
    """
    
    householdConstraints = (tract[150:166] * tract.portion).astype(int) #HOUSEHOLDS BY TYPE
    hh_cnt = pd.Series(np.zeros(11),dtype=int) #11 household types (group quarters are not household)

    # husband/wife families
    hh_cnt[0] = householdConstraints[4] - householdConstraints[5]; # husband-wife family
    hh_cnt[1] = householdConstraints[5]; # husband-wife family, OWN CHILDREN < 18
    # male householders
    hh_cnt[2] = householdConstraints[6] - householdConstraints[7]; # single male householder
    hh_cnt[3] = householdConstraints[7]; # single male householder, OWN CHILDREN < 18
    # female householders
    hh_cnt[4] = householdConstraints[8] - householdConstraints[9]; # single female householder
    hh_cnt[5] = householdConstraints[9]; # single female householder, OWN CHILDREN < 18
    # nonfamily householders
    hh_cnt[6] = householdConstraints[10] - householdConstraints[11]; # nonfamily group living
    hh_cnt[7] = householdConstraints[12] - householdConstraints[13]; # lone male < 65
    hh_cnt[8] = householdConstraints[13]; # lone male >= 65
    hh_cnt[9] = householdConstraints[14] - householdConstraints[15]; # lone female < 65
    hh_cnt[10] = householdConstraints[15]; # lone female >= 65
    return hh_cnt

# for a specific household, generate its member
def gen_households(hh_type, people, mask):
    """
    Eleven household types:
    0         h&w (no<18) husband and wife
    1      h&w&ch (ch<18) husband and wife with kids
    2        male (no<18) male with no kids (wife not present)
    3        male (ch<18) male with kids (wife not present)
    4      female (no<18) female with no kids (husband not present)
    5      female (ch<18) female with kids (husband not present)
    6     nonfamily group
    7       lone male <65 male younger than 65 lives alone 
    8       lone male >65 male older than 65 lives alone
    9      lone female<65 female younger than 65 lives alone
    10     lone female>65 female older than 65 lives alone
    """
    members = []
    
    #first, create household head
    #age range of the householder for each household type  (the range comes from dp age range above)
    head_ranges = [range(4,18), range(4,14), range(4,18), range(4,14), range(22,36), range(22,30),
            chain(range(1,18),range(19,36)), range(4,13), range(13,18), range(21,31), range(31,36)]
    '''
      meaning of the head_ranges: 
                [(15,99)/m/hh0,(20,70)/m/hh1,(15,99)/m/hh2,(20,70)/m/hh3,(15,99)/f/hh0,(20,70)/f/hh1,
                 (15,99)/f/hh4,(15,65)/f/hh5,(20,65)/m/hh7,(65,99)/m/hh8,(15,65)/f/hh9,(65,99)/f/hh10]
      head_sex= [('m'),('m'),('m'),('m'),('f'),('f'),
                 ('f'),('f'),('m'),('m'),('f'),('f')]
    
    '''
    
    #add the householder
    pot = people[mask].code #potential's age
    iindex = pot[pot.isin(head_ranges[hh_type])].index[0] #potential's age is in the range of this hh type
    h1 = people.loc[iindex] #age & sex of h1
    mask[iindex] = False
    members.append(iindex)

    #if living alone then return the members
    if hh_type > 6:
        return members

    #if husband and wife, add the wife
    if hh_type in (0,1):
        pot = people[mask]
        pot = pot[pot.age>=18] 
        
        iindex = pot.code[pot.code.isin(range(h1.code+16,h1.code+20))]
        if not iindex.empty:
            iindex=iindex.index[0]
        else:
            iindex = pot.code[pot.code.isin(range(h1.code+15,h1.code+21))]
            if not iindex.empty:
                iindex=iindex.index[0]
            else:
                iindex=pot.code[pot.code.isin(range(h1.code+13,h1.code+23))]
                if not iindex.empty:
                    iindex=iindex.index[0]
                else:
                    iindex=pot.code   #if still cannot find anyone, then randomly pick a person >18.
                    iindex=iindex.index[0]
        
        h2 = people.loc[iindex] 
        mask[iindex] = False
        members.append(iindex)

    #if may have children 18+ then add them
#         if (hh_type <= 5) and (h1.age > 36) and (np.random.random() < 0.3):
#         #to have a child older than 18, h1 should be at least 37
#             pot = people[mask]
#             iindex = pot[pot.age.isin(range(18,h1.age-15))].index[0]
#             ch18 = people.ix[iindex] #child should be at least 19 years younger than h1
#             mask[iindex] = False
#             members.append(iindex)

    """A child includes a son or daughter by birth (biological child), a stepchild,
    or an adopted child of the householder, regardless of the child’s age or marital status.
    The category excludes sons-in-law, daughters- in-law, and foster children."""
    #household types with at least one child (18-)
    if hh_type in (1,3,5):
        #https://www.census.gov/hhes/families/files/graphics/FM-3.pdf
        if hh_type == 1:
            num_of_child = max(1,abs(int(np.random.normal(2)))) #gaussian touch
        elif hh_type == 3:
            num_of_child = max(1,abs(int(np.random.normal(1.6)))) #gaussian touch
        elif hh_type == 5:
            num_of_child = max(1,abs(int(np.random.normal(1.8)))) #gaussian touch

        pot = people[mask]
        if hh_type == 1:
            iindices = pot[(pot.age<18) & (45>h2.age-pot.age)].index[:num_of_child]
        else: #father (mother) and child age difference not to exceed 50 (40)
            age_diff = 45 if hh_type == 5 else 55
            iindices = pot[(pot.age<18) & (age_diff>h1.age-pot.age)].index[:num_of_child]
        for i in iindices:
            child = people.loc[i]
            mask[i] = False
            members.append(i)

    #if nonfamily group then either friends or unmarried couples
    if hh_type == 6:
        pot = people[mask].code
        num_of_friends = max(1,abs(int(np.random.normal(1.3)))) #gaussian touch
        iindices = pot[pot.isin(range(h1.code-2,h1.code+3))].index[:num_of_friends]
        for i in iindices:
            friend = people.loc[i]
            mask[i] = False
            members.append(i)

    return members


def populate_households(tract, people, hholds):

    mask = pd.Series(True,index=people.index) #[True]*len(people)
    
    hholds['members'] = hholds.htype.apply(gen_households,args=(people, mask,))
    

    """The seven types of group quarters are categorized as institutional group quarters
    (correctional facilities for adults, juvenile facilities, nursing facilities/skilled-nursing facilities,
    and other institutional facilities) or noninstitutional group quarters (college/university student housing,
    military quarters, and other noninstitutional facilities)."""
    group_population = int(tract.DP0120014 * tract.portion) #people living in group quarters (not in households)
    #gq_indices = people[(people.age>=65) | (people.age<18)].index[:group_population]
    gq_indices = people[mask].index[:group_population]
    #for i in gq_indices: mask[i] = False
    mask.loc[gq_indices] = False

    #now distribute the remaining household guys as relatives...
    relatives = people[mask].index
    if len(relatives) != 0:
        it = iter(relatives) #sample by replacement
        relative_hhs = hholds[hholds.htype<7].sample(n=len(relatives),replace=True)
        relative_hhs.members.apply(lambda x: x.append(next(it))) #appends on mutable lists
    #for i in relatives: mask[i] = False
        mask.loc[relatives]= False
    #print('is anyone left homeless:',any(mask))
    #add those living in group quarters as all living in a house of 12th type
    if group_population > 0:
        hholds.loc[len(hholds)] = {'htype':11, 'members':gq_indices}
    
    # name households
    hholds = hholds.set_index(tract.name+'h'+pd.Series(np.arange(len(hholds)).astype(str)))

    def hh_2_people(hh,people):
        for m in hh.members:
            people.loc[m,'hhold'] = hh.name
            people.loc[m,'htype'] = hh.htype
    
    hholds.apply(hh_2_people,args=(people,),axis=1)
    people['htype'] = people.htype.astype(int)
 
   
# 3. assign workplaces 

def assign_workplaces(tract,people,od):
    """
    if the destination tract of a worker is not in our DP dataset
    then we assign his wp to 'DTIDw', otherwise 'DTIDw#'
    
    the actual size distribution of establishments is lognormal
    https://www.princeton.edu/~erossi/fsdae.pdf
    """
    #destination tracts and numbers
    td = od[od['home'] == tract.name].set_index('work').S000
    td = (td*tract.portion).apply(np.ceil).astype(int) #from this tract to others
    # 58.5%: US population (16+) - employment rate
    # https://data.bls.gov/timeseries/LNS12300000
    
    if td.sum()<= len(people[people.age>=18]): 
        employed = people[people.age>=18].sample(td.sum()).index #get the employed
        dtract = pd.Series(np.repeat(td.index.values, td.values))
        
    else:
        employed=people[people.age>=18].index
        dtract = pd.Series(np.repeat(td.index.values, td.values))
        dtract=dtract.sample(len(employed))
    
     
    people.loc[employed,'wp'] = dtract.apply(lambda x: x+'w'+str(np.random.choice(dp.loc[x,'WP_CNT'],p=dp.loc[x,'WP_PROBA'])) if x in dp.index else x+'w').values
    
# 4. assign students to school/daycare

def assign_students_to_schools(tract,people,school,buffer=0.3):
    """
    Get the schools within 30km that accepts students at this age.
    loop over schools from nearest to farest:
      if school has any space then enroll
    if no school available then
      enroll to the school with the lowest enrollment rate
    update the enrollment of the school
    PERCENTAGE ERRORS 
    """
    def assign_school(x,pot,school):
        sch = pot.distance(x.geometry).sort_values()
        for s in sch.index: #from nearest to farest
            if school.loc[s,'current'] < school.loc[s,'ENROLLMENT']:
                school.loc[s,'current'] += 1
                return s

        #if no space left at any school within the buffer
        least_crowded = (pot.current/pot.ENROLLMENT).idxmin()
        school.loc[least_crowded,'current'] += 1
        return least_crowded

    kids = people.age<18
    buff = tract.geometry.buffer(buffer)
    
    sch_pot = school[school.intersects(buff)] #filter potential schools and daycares
    people.loc[kids,'wp'] = people[kids].apply(assign_school,args=(sch_pot,school),axis=1)
    
# 5. get errors
def get_errors(tract,people):
    """Percentage errors
    """
    err = {}    
    portion = tract.portion # what portion of the tract is included
    senior_actual = int(tract.DP0150001 * portion) # Households with individuals 65 years and over
    minor_actual = int(tract.DP0140001 * portion) # Households with individuals under 18 years

    err['population'] = tract.DP0010001
    err['in_gq'] = tract.DP0120014
    
    #some of the following data == 0, so if statements are added, if data doesn't exist, error = None
    if tract.DP0170001 != 0:
        avg_synthetic_family = people[people.htype<6].groupby('hhold').size().mean()
        err['avg_family'] = 100*(avg_synthetic_family - tract.DP0170001) / tract.DP0170001
    else: 
        err['avg_family'] = None 
    if tract.DP0160001 !=0:
        err['avg_hh'] = 100*(people[people.htype!=11].groupby('hhold').size().mean() - tract.DP0160001) / tract.DP0160001 
    else:
        err['avg_hh'] = None
    if senior_actual != 0:
        err['senior_hh'] = 100*(people[people.age>=65].hhold.nunique() - senior_actual) / senior_actual
    else:
        err['senior_hh']= None
    if minor_actual !=0:
        err['minor_hh'] = 100*(people[people.age<18].hhold.nunique() - minor_actual) / minor_actual
    else:
        err['minor_hh']= None
    return pd.Series(err,name=tract.name)

# create space
#shapely geometries are not hashable, here is my hash function to check the duplicates
def hash_geom(x):
    if x.geom_type == 'MultiLineString':
        return tuple((round(lat,6),round(lon,6)) for s in x for lat,lon in s.coords[:])    
    else:
        return tuple((round(lat,6),round(lon,6)) for lat,lon in x.coords[:])
    
# create spaces
def create_spaces(tract, hcnt, od, road, HD=0.0005, WD=0.0002):
    portion = tract.portion# what portion of the tract is included
    # create houses
    # DP0180001: Total housing units, DP0180002: Occupied housing units
    # hcnt = int(tract.DP0180002 * portion) #number of households DP0130001 == DP0180002
    if tract.DP0120014 > 0: hcnt += 1 #people living in group quarters (not in households)
    mask = road.intersects(tract.geometry) 
    hmask = mask & road.MTFCC.str.contains('S1400|S1740')
    hrd = road[hmask].intersection(tract.geometry)
    hrd = hrd[hrd.geom_type.isin(['LinearRing', 'LineString', 'MultiLineString'])]
    hrd = hrd[~hrd.apply(hash_geom).duplicated()]
    houses = hrd.apply(lambda x: pd.Series([x.interpolate(seg) for seg in np.arange(HD,x.length,HD)]))
    houses = houses.unstack().dropna().reset_index(drop=True) #flatten
    houses = houses.sample(n=hcnt,replace=True).reset_index(drop=True)
    houses.index = tract.name + 'h' + houses.index.to_series().astype(str)
    #create workplaces
    #jcnt = int(portion * od[od.work==tract.name].S000.sum() / avg_wp)
    wmask = mask & road.MTFCC.str.contains('S1200')
    wrd = road[wmask].intersection(tract.geometry)
    wrd = wrd[wrd.geom_type.isin(['LinearRing', 'LineString', 'MultiLineString'])]
    wrd = wrd[~wrd.apply(hash_geom).duplicated()]
    #workplaces on S1200
    swps = wrd.apply(lambda x: pd.Series([x.interpolate(seg) for seg in np.arange(x.length,WD)]))
    #workplaces on the joints of S1400|S1740
    rwps = hrd.apply(lambda x: Point(x.coords[0]) if type(x) != MultiLineString else Point(x[0].coords[0]))
    if len(swps) > 0:
        wps = rwps.append(swps.unstack().dropna().reset_index(drop=True))
    else:
        wps=rwps
    wps = wps.sample(n=tract.WP_CNT,replace=True).reset_index(drop=True)
    wps.index = tract.name + 'w' + wps.index.to_series().astype(str)
    return gpd.GeoSeries(houses), gpd.GeoSeries(wps)

def create_networks(people,k=4,p=.3):
    g = nx.Graph()
    g.add_nodes_from(people.index)
    grouped = people.groupby('hhold')
    grouped.apply(lambda x: create_edges(x,g,etype='hhold',k=k,p=p))
    grouped = people[people.age>=18].groupby('wp')
    grouped.apply(lambda x: create_edges(x,g,etype='work',k=k,p=p))
    grouped = people[people.age<18].groupby('wp')
    grouped.apply(lambda x: create_edges(x,g,etype='school',k=k,p=p))
    return g #max(nx.connected_component_subgraphs(g), key=len)


def create_edges(x,g,etype=None,k=4,p=.3):
    """Creates the edges within group `g` and adds them to `edges`
    
    if the group size is <=5, then a complete network is generated. Otherwise,
    a Newman–Watts–Strogatz small-world graph is generated with k=4 and p=0.3
    
    http://www.sciencedirect.com/science/article/pii/S0375960199007574
    """
    if len(x)<=5:
        sw = nx.complete_graph(len(x))
    else:
        sw = nx.newman_watts_strogatz_graph(len(x),k,p)
    sw = nx.relabel_nodes(sw,dict(zip(sw.nodes(), x.index.values)))
    if etype:
        g.add_edges_from(sw.edges(),etype=etype)
    else:
        g.add_edges_from(sw.edges())
    
def synthesize(tract, od, road, school, errors, population, wps):
    start_time = timeit.default_timer()
    print(tract.name,'started...',end=' ')

    try:
        people = create_individuals(tract)
        create_households(tract,people)
        assign_workplaces(tract,people,od)
        houses, wp = create_spaces(tract, people.hhold.nunique(), od, road)
        people['geometry'] = people.hhold.map(houses)
        assign_students_to_schools(tract,people,school)
        err = get_errors(tract,people)
        wps.append(wp)
        population.append(people)
        errors.append(err)
    except:
        print(tract.name," has problems")
        fd = open('../output/problematic_tracts2.csv','a')
        fd.write(tract.name)
        fd.close()
    print(tract.name,'now ended ({:.1f} secs)'.format(timeit.default_timer() - start_time))





population = []
errors = []
wps = []


# import problematic tracts
prob=pd.read_csv("../output/final_prob.csv",header=None,dtype=str)
prob.drop(1,axis=1,inplace=True)
data=dp[dp.index.isin(prob[0])]

data.apply(lambda t: synthesize(t,od,road,school,errors, population, wps),axis=1);

with open('../output/'+"errors"+"_prob_1"+".pkl", 'wb') as f:
    pickle.dump(errors, f)
with open('../output/'+"population"+"_prob_1"+".pkl", 'wb') as f:
    pickle.dump(population, f)
with open('../output/'+"wps"+"_prob_1"+".pkl", 'wb') as f:
    pickle.dump(wps, f)
  




