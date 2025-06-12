
Task: Cluster POIs based on Foursquare places with finely grained dataset
Goal: Extract POI-POI and POI-Temporal behavior patterns using temporal and semantic features.


2. Convert timestamps to hour-of-week (0–167) and build a 168-dim temporal activity vector per user

TODO TASK : create a temporal feature extraction script for extracting the temporal features for each user, also get the temporal entropy here.

3. For each user, capture the spatial activity 

TODO TASK : create a spatial feature extraction acript for extracting teh spatial features for each user. The feature is POI primary category hitogram (top 20 categories), then spatial entropy based on POI visits



For the below subsequent steps, use a master script say - user_cluster .csv along with  the visual plots

4. Normalize all the spatial and the temporal features per user

5. Use cosine similarity or KL-divergence for pairwise distance
6. Cluster users using Agglomerative Clustering (or Spectral)
7. Visualize cluster heatmaps and project features to 2D using t-SNE
