# Ideas 

### User activity graph embedding
#### Paper - Adversarial Substructured Representation Learning for Mobile User Profiling
#### Paper link - https://par.nsf.gov/servlets/purl/10174829 

##### Excerpt:
an adversarial sub- structured learning framework for mobile user profiling. 
Specifically, our contributions are as follows: (1) We create user activity
graphs to describe the characteristics, patterns, and preferences of
mobile users. (2) We reformulate mobile user profiling as a problem of 
learning deep representations of user activity graphs. (3)
We identify another structure information: substructures in user
activity graphs and develop an adversarial substructured learning
paradigm, including an auto-encoder, a detector, and an adversarial
trainer, to preserve both the entire graph and substructure infor-
mation. (4) We pre-train a convolutional neural network (CNN) to
approximate traditional subgraph detection algorithms to solve the
non-differentiable issue. (5) We apply the user profiling results to
the application of next activity type prediction, and present exten-
sive experiments to demonstrate the enhanced performance of the
proposed method with real-world mobile checkin data.

 - Subgraph analysis: we focus on two types of substructures: 
    - (1) high-frequency vertexes, of which the cumulative visit frequency is above the pre-defined threshold;
    - (2) circles. 
    
    - Specifically, a high-frequency vertex in a user activity graph represents the personalized preference for a specific type of activities; a circle in a user activity graph represents the personalized preference for a close-loop consecutive activity pattern. Both high-frequency vertexes and circles can imply the unique activity patterns of a user in his/her daily life.


- To conduct robustness check, we apply our method to different
subgroups of data to examine the variance of our performances.
Specifically, we equally split the dataset into five time periods,
including (1) 12 Apr. 2012 – 12 Jun. 2012, (2) 13 Jun. 2012 – 13 Aug.
2012, (3) 14 Aug. 2012 – 14 Oct. 2012, (4) 15 Aug. 2012 – 15 Oct. 2012,
and (5) 16 Oct. 2012 – 16 Feb. 2013.

- Specifically, we denote (1) StructRL-Node: a variant of our frame-
work that only consider discrete vertexes substructure; (2) StructRL-
Circle: a variant of our framework that only consider as circle sub-
structure; (3) StructRL: our proposed method that consider both.
Figure 7 shows the performances of StructRL-Circle always slightly
outperforms StructRL-Node; in other words, the substructure of cir-
cle is more effective than the substructure of discrete vertexes for
describing user activity patterns. The substructure of circle shows a
user’s circle activity transition patterns, while the substructure of
independent vertexes shows some independent POI categories that
users highly and intensively prefer. Therefore, the substructure of
circle can describe the correlations among POI categories to imply a
user’s particular lifestyle patterns, which is more informative than
the substructure of independent vertexes.


##### user profiling 
- User Profiling User profiling refers to the efforts of extracting
a user’s interest and behavioral patterns from users’ activities. Gen-
erally speaking, the user profiling techniques can be categorized
into two groups: (1) static profiling and (2) dynamic profiling. Static
profiling refers to learn representations of users based on the ag-
gregated dataset that depends on the temporal perspective [ 11 ].
For example, Farseev et al. proposed to learn user profile via in-
tegration of multiple data sources from Twitter, Foursquare and
Instagram [ 13]. Farseev et al. proposed multi-source individual user
profile learning framework named “TweetFit” that can handle data
incompleteness and perform wellness attributes inference from sen-
sor and social media data simultaneously [ 12 ]. On the other hand,
dynamic profiling refers to modeling user representations consid-
ering the temporal effects that user profiles may change over time.
Akbari et al. proposed an approach which directly learns the em-
bedding from longitudinal data of users that simultaneously learns
a low-dimensional latent space as well as the temporal evolution
of users in the wellness space [ 1]. Du et al. proposed a framework
which connects self-exciting point processes and low-rank models
to capture the recurrent temporal patterns in a large collection of
user-item consumption pairs [ 10 ]. Zhao et al. proposed a spatial-
temporal latent ranking (STELLAR) method that capture the impact
of time on successive POI recommendation [ 50 ]. Xiao et al. pro-
posed to quantify user influence from user interactions in social
networks tp explain price stock [44].


#### Paper - Context-Aware Point-of-Interest Recommendation Based on Similar User Clustering and Tensor Factorization

#### Paper link - analysis_set3/paper/ijgi-12-00145.pdf

##### Excerpt:

a personalized POI recommendation method named CULT-TF, which
incorporates similar users’ contextual information into the tensor factorization model, is
proposed in this paper. The following contributions are provided by this work:
1. We define a user activity model and a user similarity model that can integrate contextual
information of users’ check-in behavior to calculate user activity and user similarity;
2. A similar user clustering method based on user activity and user similarity is presented
to select the most influential active users as clustering centers based on user activity and
to cluster users into several similar user clusters according to user similarity;
3. A U-L-T tensor that incorporates contextual information using user activity, POI
popularity, and time slot popularity as the eigenvalues in the U, L, and T dimensions,
which improves the integration of contextual information, is presented;
4. The CULT-TF recommendation method based on tensor factorization, which decreases
the complexity of the matrix-integrating rich contextual information by clustering
similar users, clustering POIs into regions of interest (ROIs), and encoding check-
in timestamps to time slots, to realize the reduction of the U, L, and T dimension,
respectively, is proposed. In this way, CULT-TF reduces the complexity of the recom-
mendation matrix while integrating the richness of the contextual information.

- similar user clustering :

The core of similar user clustering consists of two parts: selecting clustering centers
and calculating user similarity. In this paper, we select several users with the highest
activity as clustering centers based on their social data and check-in data, calculate the user
similarity based on their social relationship similarity and check-in behavior similarity, and
then cluster the remaining users in the clustering center with the highest user similarity, as
detailed in Section 4.1

- modelling
Unlike the POI recommendation method, which directly uses the user, location, and time
information of check-in data to construct the tensor model, our proposed method models
user activity, POI popularity, and time slot popularity as the eigenvalues of the U, L, and T
dimensions. 

- TF 
Tensor factorization can
be used to present and maintain the structural characteristics of high-dimensional data by
mapping the relationships in the original space to a low-dimensional space and extracting
the potential relation between different dimensions to calculate the approximate tensor
of the missing values and compensate for the matrix sparsity problem caused by missing
data.