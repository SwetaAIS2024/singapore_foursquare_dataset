### Code explanation 

SwetaAIS2024
we also need to show the points - like who is the user and POI type or category in that cluster - may be another plot or something

GitHub Copilot
To visualize which users belong to which cluster and highlight their dominant POI type/category, I will:

For each user, determine their dominant POI type (the POI type with the highest total across all their POI columns).
Create a scatter plot (PCA 2D) where each point is a user, colored by cluster, and annotated (or marked) by their dominant POI type.
Optionally, provide a legend for the top N POI types and/or output a CSV mapping user_id, cluster, and dominant POI type.
This will help you see not just the cluster, but also the main POI type for each user in the plot.

I will update the script to add this functionality.

cluster_users_by_activity_trends.py+51-0
Now, the script will:

Compute each user's dominant POI type.
Plot clusters in PCA space, using marker shapes to indicate the user's dominant POI type (top 10 types get unique markers).
Save this plot as user_clusters_pca_dominant_poi_plot.png.
Output a CSV (user_cluster_dominant_poi.csv) mapping user_id, cluster, and dominant POI type.
This will help you visually and analytically explore which users and POI types are in each cluster. Let me know if you want further customization!


