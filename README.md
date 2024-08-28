# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python for weather classification.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)
Load the dataset
Calculate the linear trend values using least square method
Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/content/weather_classification_data.csv'
df = pd.read_csv(file_path)

# Assuming the data is in a column named 'Temperature'
y = df['Temperature'].values
x = np.arange(len(y))

# A - Linear Trend Estimation
coeffs_linear = np.polyfit(x, y, 1)  # 1st degree polynomial (linear)
linear_trend = np.polyval(coeffs_linear, x)

# Plot Linear Trend
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Original Data', color='blue')
plt.plot(x, linear_trend, label='Linear Trend', color='red')
plt.legend()
plt.title('Linear Trend Estimation')
plt.xlabel('Index')
plt.ylabel('Temperature')
plt.savefig('linear_trend_estimation.png')
plt.show()

# B - Polynomial Trend Estimation
coeffs_poly = np.polyfit(x, y, 2)  # 2nd degree polynomial
poly_trend = np.polyval(coeffs_poly, x)

# Plot Polynomial Trend
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Original Data', color='blue')
plt.plot(x, poly_trend, label='Polynomial Trend', color='green')
plt.legend()
plt.title('Polynomial Trend Estimation')
plt.xlabel('Index')
plt.ylabel('Temperature')
plt.savefig('polynomial_trend_estimation.png')
plt.show()
```


### OUTPUT

# A - LINEAR TREND ESTIMATION

![Screenshot 2024-08-28 152923](https://github.com/user-attachments/assets/95527308-9e64-40a2-be28-f0b5332d88cc)


# B- POLYNOMIAL TREND ESTIMATION

![Screenshot 2024-08-28 152955](https://github.com/user-attachments/assets/797fec24-8de1-45c3-9155-af4f16cd993d)

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion for weather classification  has been executed successfully.
