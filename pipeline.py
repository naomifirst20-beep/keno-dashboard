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

    if len(raw_data) < 6:
        raise ValueError("🚫 Not enough draw data to train the model. Add at least 6 rows to data/draws.csv.")

    X_multi, y_multi = multi_draw_features(raw_data)
    X_freq = frequency_features(raw_data)
    cluster_ids = cluster_labels(raw_data)

    # Align row counts
    min_len = min(len(X_multi), len(X_freq), len(cluster_ids.iloc[:-5]))
    X_multi = X_multi.iloc[:min_len]
    X_freq = X_freq.iloc[:min_len]
    cluster_ids = cluster_ids.iloc[:min_len]

    # Combine features
    X_combined = pd.concat([X_multi, X_freq, cluster_ids], axis=1)

    # Clean feature names and types
    X_combined.columns = [f"f{i}" for i in range(X_combined.shape[1])]
    X_combined = X_combined.apply(pd.to_numeric, errors='coerce')
    X_combined = X_combined.dropna(axis=1)  # Drop any columns with NaNs
    X_combined = X_combined.reset_index(drop=True)

    y_combined = y_multi.iloc[:min_len].reset_index(drop=True)

    # Debug logging
    print("🔍 Debug Info:")
    print("X_combined shape:", X_combined.shape)
    print("y_combined shape:", y_combined.shape)
    print("X_combined types:\n", X_combined.dtypes)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_combined, y_combined)
    return model, X_combined, y_combined

def predict_next(model, recent_draws, window=5):
    # Apply same feature engineering
    X_multi = pd.DataFrame([recent_draws.tail(window).values.flatten()])
    X_freq = pd.DataFrame([recent_draws.tail(window).sum().tolist()])
    cluster_id = KMeans(n_clusters=5, random_state=42).fit_predict(recent_draws)[-1]
    cluster_df = pd.DataFrame([[cluster_id]], columns=["cluster_id"])

    # Combine and clean
    X_input = pd.concat([X_multi, X_freq, cluster_df], axis=1)
    X_input.columns = [f"f{i}" for i in range(X_input.shape[1])]
    X_input = X_input.apply(pd.to_numeric, errors='coerce').dropna(axis=1).reset_index(drop=True)

    prediction = model.predict(X_input)[0]
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




