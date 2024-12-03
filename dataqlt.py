import pandas as pd
file_path = "I:/online_retail_II.xlsx"
df_2009_2010 = pd.read_excel(file_path, sheet_name = "Year 2009-2010")
df_2010_2011 = pd.read_excel(file_path, sheet_name = "Year 2010-2011")

#print(df.head())
#print(df.info())
#print(df.describe(include = "all"))
# Kiểm tra tổng số giá trị bị thiếu trong mỗi cột
#missing_values = df.isnull().sum()
#print("Số giá trị bị thiếu theo cột:\n", missing_values)
# Kiểm tra tỷ lệ giá trị bị thiếu
#missing_percentage = (missing_values / len(df)) * 100
#print("Tỷ lệ dữ liệu bị thiếu: ", missing_percentage)
df_2009_2010 = df_2009_2010.dropna(subset = ['Customer ID'])
df_20010_2011 = df_2010_2011.dropna(subset = ['Customer ID'])
df_2009_2010['Description'] = df_2009_2010['Description'].fillna('Unknow')
df_2010_2011['Description'] = df_2010_2011['Description'].fillna('Unknow')
duplicates_2009_2010 = df_2009_2010.duplicated()
df_2009_2010 = df_2009_2010.drop_duplicates()
duplicates_2010_2011 = df_2010_2011.duplicated()
df_2010_2011 = df_2010_2011.drop_duplicates()
df_2009_2010 = df_2009_2010[(df_2009_2010['Quantity'] > 0) & (df_2009_2010['Price'] > 0)]
df_2010_2011 = df_2010_2011[(df_2010_2011['Quantity'] > 0) & (df_2010_2011['Price'] > 0)]
unique_customers_2009_2010 = df_2009_2010['Customer ID'].nunique()
unique_customers_2010_2011 = df_2010_2011['Customer ID'].nunique()
customer_frequency_2009_2010 = df_2009_2010['Customer ID'].value_counts()
customer_frequency_2010_2011 = df_20010_2011['Customer ID'].value_counts()
#Thêm cột Frequency
df_2009_2010['Frequency'] = df_2009_2010['Customer ID'].map(customer_frequency_2009_2010)
df_2010_2011['Frequency'] = df_2010_2011['Customer ID'].map(customer_frequency_2010_2011)
df_2009_2010['Revenue'] = df_2009_2010['Quantity'] * df_2009_2010['Price']
df_2010_2011['Revenue'] = df_2010_2011['Quantity'] * df_2010_2011['Price']
# Sắp xếp lại thứ tự các cột (nếu cần)
columns_order_2009_2010 = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'Price', 
                 'Customer ID', 'Country', 'Revenue', 'Frequency']
df_2009_2010 = df_2009_2010[columns_order_2009_2010]
columns_order_2010_2011 = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'Price', 
                 'Customer ID', 'Country', 'Revenue', 'Frequency']
df_2010_2011 = df_2010_2011[columns_order_2010_2011]
output_file = "fixed_online_retail_II.xlsx"
with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
    # Lưu sheet 'Year 2009-2010'
    df_2009_2010.to_excel(writer, sheet_name='Year 2009-2010', index=False)
    
    # Lưu sheet 'Year 2010-2011'
    df_2010_2011.to_excel(writer, sheet_name='Year 2010-2011', index=False)








