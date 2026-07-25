import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler



dataset_path = r"C:\Users\HARIKRISHNAN\.cache\kagglehub\datasets\keplersmachines\kepler-labelled-time-series-data\versions\3"


def load_data():
    train = pd.read_csv(f"{dataset_path}\exoTrain.csv")
    test = pd.read_csv(f"{dataset_path}/exoTest.csv")

    x_test = test.iloc[:, 1:]
    y_test = test.iloc[:, 0] - 1


    x_train = train.iloc[:, 1:]    # light curve values 
    y_train = train.iloc[:, 0] - 1  # wheather it has an exoplanet or not


    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        x_scaled,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    X_train = X_train.reshape(-1, X_train.shape[1], 1)
    X_val = X_val.reshape(-1, X_val.shape[1], 1)
    x_test_scaled = x_test_scaled.reshape(-1, x_test_scaled.shape[1], 1)

    print(X_train.shape)
    print(X_val.shape)
    print(x_test_scaled.shape)

    # ros = RandomOverSampler(random_state=42)
    # X_train, y_train = ros.fit_resample(X_train, y_train)

    # return X_train, X_val, y_train, y_val
    
    return X_train, X_val, x_test_scaled, y_train, y_val, y_test
