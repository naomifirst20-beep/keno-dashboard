import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

def load_draw_data(filepath):
    df = pd.read_csv(filepath, header=None)
    binary_draws = []
    for _, row in df.iterrows():
        vector = [1 if i in row.values else 0 for i in range(1, 81)]
        binary_draws.append(vector)
    return pd.DataFrame(binary_draws)

def multi_draw_features(data, window=5):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data.iloc[i:i+window].values.flatten())
        y.append(data.iloc[i+window].values)
    return pd.DataFrame(X), pd.DataFrame(y)

def frequency_features(data, window=5):
    freq_X = []
    for i in range(len(data) - window):
        freq = data.iloc[i:i+window].sum().tolist()
        freq_X.append(freq)
    return pd.DataFrame(freq_X)

def cluster_labels(data, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return pd.DataFrame(labels, columns=["cluster_id"])

def run_pipeline(filepath):
    raw_data = load_draw_data(filepath)
    # Safety check: ensure enough data for training
    if len(raw_data) < 6:
        raise ValueError(
            "🚫 Not enough draw data to train the model. "
            "Please ensure data/draws.csv contains at least 6 rows of valid Keno draws."
        )
    X_multi, y_multi = multi_draw_features(raw_data)
    X_freq = frequency_features(raw_data)
    cluster_ids = cluster_labels(raw_data)

    X_combined = pd.concat([X_multi, X_freq, cluster_ids.iloc[:-5]], axis=1)
    y_combined = y_multi

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_combined, y_combined)
    return model, X_combined, y_combined

def predict_next(model, recent_draws, window=5):
    input_vector = recent_draws.tail(window).values.flatten().reshape(1, -1)
    prediction = model.predict(input_vector)[0]
    return [i+1 for i, val in enumerate(prediction) if val == 1]

def build_leaderboard(model, data, window=5):
    leaderboard = []
    for i in range(len(data) - window - 1):
        input_draws = data.iloc[i:i+window]
        actual_draw = data.iloc[i+window]
        predicted = predict_next(model, input_draws)
        actual = [j+1 for j, val in enumerate(actual_draw.tolist()) if val == 1]
        match = set(predicted) & set(actual)
        leaderboard.append({
            "Draw Index": i + window,
            "Predicted": predicted,
            "Actual": actual,
            "Matched": sorted(match),
            "Match Count": len(match)
        })
    return pd.DataFrame(leaderboard)

