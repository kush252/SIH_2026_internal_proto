import pandas as pd
df = pd.read_csv('d:/Kush/2nd Year/Hackathons/SIH/data/roofnet/roofnet_metadata.csv')
print('Total:', len(df))
print('Classes:\n', df['material_class'].value_counts())
print('Splits:\n', df['split'].value_counts())
