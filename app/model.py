# import necessary libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# import dataset
data=pd.read_csv('../notebook/banglore.csv')

# fill na values in 'location' column
data['location']=data['location'].fillna('Other')

# remove locations with less than 10 occurrences
loc_count=data['location'].value_counts()
data['location']=data['location'].apply(lambda x: 'other' if loc_count[x]<=10 else x)
data['location']=data['location'].str.lower()

# store unique location names for handling unseen locations during prediction
loc_name=data['location'].unique()

# extract bhk from size in integer format
data.dropna(subset=['size'],inplace=True)
data['size'] = data['size'].apply(lambda x: int(x.split(' ')[0]))

# convert availability to ready to move or not
def convert(x):
    if x=='Ready To Move':
        return x
    else:
        return 'Not ready'
data['availability']=data['availability'].apply(convert)
data['availability'].unique()

# drop society column
data.drop('society',axis=1,inplace=True)

# convert total_sqft to numerical values
def convert_sqft_to_num(x):
    if '-' in x:
        tokens = x.split('-')
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None
data['total_sqft'] = data['total_sqft'].apply(convert_sqft_to_num)

# fill na values with median
data['total_sqft']=data['total_sqft'].fillna(data['total_sqft'].median())
data['bath']=data['bath'].fillna(data['bath'].median())
data['balcony']=data['balcony'].fillna(data['balcony'].median())

# remove outliers based on price per sqft
# price*100000 because price is in lakhs
data['price_per_sqft']=data['price']*100000/data['total_sqft']
data=data[data['price_per_sqft']<data['price_per_sqft'].quantile(0.97)]
data=data[data['price_per_sqft']>data['price_per_sqft'].quantile(0.03)]
data.drop('price_per_sqft',axis=1,inplace=True)

# remove outliers based on total_sqft per bhk
data=data[data['total_sqft']/data['size']>=250]

x=data.drop('price',axis=1)
y=data['price']

cat_cols=['area_type','availability','location']

cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', cat_transformer, cat_cols)
    ], remainder='passthrough'
)

model=Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=300,max_depth=None,min_samples_leaf=1,random_state=42))
])

model.fit(x,y)
joblib.dump(model,'../model/bangalore_house_price_model.pkl')
joblib.dump(loc_name,'../model/loc_name.pkl')