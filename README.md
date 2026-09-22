# Moosic-DBSCAN-Playlist-Clustering
Clustering ~5,000 Spotify tracks with DBSCAN to prototype automatic playlist generation for Moosic — using PCA for feature reduction, a k-distance graph for eps selection, and t-SNE + real playlist comparisons to validate cluster quality.

# Moosic – DBSCAN Playlist Clustering

> *A prototype exploring whether Spotify audio features can automatically group songs into human-recognisable playlists.*

## Business Context

Moosic is a growing music start-up that curates mood- and style-based playlists by hand. As demand grows, their team of music experts can't keep up — so Moosic brought in Data Science to explore whether **audio features alone can identify "similar songs"** the way a human curator would, and to test which clustering approach works best for automatic playlist generation.

This repository covers **my part of the project: clustering ~5,000 songs using DBSCAN**, based on audio features pulled from the Spotify API (tempo, energy, danceability, acousticness, etc.).

**Key questions explored:**
- Can Spotify's audio features capture the "mood" or "style" of a song well enough to group similar tracks together?
- Is DBSCAN a good fit for this task compared to alternatives like K-Means or Agglomerative Clustering?

## Approach

### 1. Dimensionality Reduction (PCA)
Several audio features overlapped heavily — energy, loudness, and acousticness shared up to 85% of their signal. Instead of manually dropping one, PCA was used to merge this redundancy automatically. **3 principal components captured 95% of the variance** in the original 9 features, giving a cleaner, less biased feature space for clustering.

![PCA Explained Variance](images/pca_explained_variance.png)

### 2. Choosing `eps` for DBSCAN
A **k-distance graph** (distance to each point's 3rd nearest neighbour, sorted) was used to identify a reasonable `eps` range — the "elbow" hints at where density changes sharply.

To pick a precise value, DBSCAN was run across a sweep of `eps` values, tracking the resulting number of clusters and the noise percentage:

| eps  | n_clusters | noise_% |
|------|-----------|---------|
| 0.05 | 47        | 83.4    |
| 0.10 | 137       | 47.1    |
| 0.20 | 52        | 11.0    |
| 0.30 | 30        | 3.3     |
| 0.40 | 26        | 1.3     |
| **0.45** | **25**    | **0.9**     |
| 0.50 | 25        | 0.5     |
| 0.60 | 25        | 0.3     |

`eps = 0.45` was chosen as the point where the cluster count stabilises at 25 with very low noise (<1%) — increasing `eps` further only reduces noise slightly without changing the clustering structure.

![k-distance graph and eps sweep](images/eps_selection.png)

### 3. Final Model

```python
dbscan = DBSCAN(eps=0.45, min_samples=5)
dbscan.fit(scaled_features_df)
```

**Result: 25 clusters, 68 outliers** (songs too sparse/unique to fit any cluster).

### 4. Visualisation (t-SNE)
The high-dimensional clusters were projected to 2D with t-SNE to visually sanity-check the separation between clusters.

![t-SNE visualisation of DBSCAN clusters](images/tsne_clusters.png)

The clusters are visually distinct and well-separated, suggesting DBSCAN is finding real structure in the audio feature space — not arbitrary splits.

### 5. Qualitative Validation
To check whether the clusters actually mean something musically, songs from real Moosic playlists were mapped back to their assigned clusters:

![Playlist samples mapped to clusters](images/playlist_samples.png)

- **"Chilling in Vienna"** → dominated by classical composers (Bach, Chopin, Schubert, Mozart)
- **"Chilling in Berlin"** → dominated by a distinct alt-rock/indie cluster
- **"Afternoons in Summer"** → dominated by Brazilian jazz / bossa nova artists (Stan Getz, Toots Thielemans, Caetano Veloso)

This is a strong sign that the audio-feature-based clusters correspond to genuinely recognisable musical styles, not just statistical noise.

## Findings

- **Audio features do carry meaningful signal.** Songs that a human would group together (by genre/mood) tend to land in the same DBSCAN cluster.
- **DBSCAN's strengths for this task:** it doesn't require specifying the number of clusters upfront, it naturally isolates "unique" songs as noise/outliers instead of forcing them into a bad-fit cluster, and it can find non-spherical clusters (useful since musical style isn't necessarily a simple round blob in feature space).
- **DBSCAN's weaknesses:** sensitive to the `eps` parameter (needs tuning per dataset), and it doesn't scale as smoothly as K-Means on very large catalogs since it relies on density estimation across all points. It also may treat legitimately valid "rare style" songs as noise rather than a small cluster.

## Next Steps

This is an early-stage prototype. Future work could include:
- Comparing DBSCAN results head-to-head against K-Means and Agglomerative Clustering on the same feature set
- Tuning `min_samples` alongside `eps` for a finer-grained sweep
- Getting qualitative feedback from Moosic's music experts on cluster quality at scale

