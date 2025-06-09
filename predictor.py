import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Fetching Game Data (Example)
data = {'Time': ['22:00', '22:10', '22:20', '22:30'],
        'Multiplier': [1.5, 2.1, 1.8, 2.4]} 

df = pd.DataFrame(data)
df.to_csv("aviator_data.csv", index=False)

# Train Machine Learning Model
X = np.arange(len(df)).reshape(-1, 1)
y = df['Multiplier']

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# Predict Next Multiplier
next_prediction = model.predict([[len(df) + 1]])
print(f"Predicted Next Multiplier: {next_prediction[0]:.2f}")
