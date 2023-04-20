## This is the r script for Yue Liu's semester project

############ Before you start ####################
## attention please!!!! 
# Because my project is analyzing bird sound, my raw data files are audio files.
# the package "WarbleR" i am using to analysis audio files can only work normally with R version 4.1.0
# So, Make sure your R version is 4.1.0 
# Check your current r version 
R.version.string
# if your R version is not 4.1.0
# for mac user, paste this link in your web browser to download R 4.1.0: cran.r-project.org/bin/macosx/base/R-4.1.0.pkg
# for windows user, visit this link: https://cran.r-project.org/bin/windows/base/old/4.1.0/



## clear the R work space 
rm(list = ls())


## set work path
getwd()
# for myself
setwd("/Users/liuyue/Desktop/Analysis4EcoData/Analysis4EcoData/SemesterProject")
# for peer review
setwd("paste where you saved my data")
# for TA
setwd("/Users/marthadrake/Desktop/Ento stats")


## load packages required by my project 
# if you have not installed the following packages, please install first
library(parallel)
library(vegan)
library(bioacoustics)
library(warbleR)
library(ggplot2)
library(Rraven)
library(cluster)
library(factoextra)
library(FactoMineR)
library(mclust)
library(Rtsne)
library(randomForest)
library(MASS)
library(fossil)
library(pbapply)
library(adehabitatHR)
library(caret)
library(Sim.DiffProc)
library(data.table)
library(tidyverse)
library(tuneR)


############## Analysis begins here ########################

## ----------------- Section One---------------------------- 
## import sound files and knowing these data

# Set the path to the folder containing all sound files we are going to use
# you can download those data files from my GitHub repo: https://github.com/YueLiuUFPU/FNR642_SemesterProject
# all audio files i am using could be downloaded from the folder "ChickdeeCalls"
sound_folder <- "./Note_d" # The "." at the beginning of "./Note_d" represents the current working directory, so this line of code specifies a folder called Note_D that is located in the current working directory.

# get wave file info 
wi <- info_wavs()
summary(wi)

# Get a list of the sound file names, and create a vector called file_names that contains the names of all files in the sound_folder directory that match the pattern "\\.wav$".
file_names <- list.files(sound_folder, pattern = "\\.wav$", full.names = FALSE) #  "\\.wav$" specifies that the file names should end with the string ".wav".
file_names
length(file_names) # 25 sound files in total

# Get a list of the sound files from location "CAD", and count the number
location_CAD <- list.files(sound_folder, pattern = "CAD",full.names = FALSE)
length(location_CAD) # 16 sound files from CAD (i.e, study site named "Martell")

# Get a list of the sound files from location "CAD", and cound the number
location_ROS <- list.files(sound_folder, pattern = "ROS",full.names = FALSE)
length(location_ROS ) # 9 sound files from ROS (i.e, study site named "Ross nature reserve")


## ----------------- Section two---------------------------- 
## transform and save the .wav files into a new format the warbleR package needs

# Set sound analysis parameters for the warbleR package
warbleR_options(wav.path = sound_folder, wl = 512, flim = c(0, 15), ovlp = 75, bp = c(1,15), parallel = parallel::detectCores() - 1)

# Now we'll put the wav clips into an extended selection table, the format the warbleR package needs
# The RDS format is a binary format used for saving R objects, and it allows for efficient reading and writing of large data sets. 
est.file.name<-file.path(sound_folder,"Extended selection table.RDS") 

# create extended selection table that summarizes the acoustic parameters of the sound files in the sound_folder directory. 
est <- selection_table(whole.recs = T, extended = T, confirm.extended = F)

# Homogenize sampling rate of sound files
# please do this because our sound files were not created with the same sampling rate during recording
est <- resample_est_waves(est)

# save  extended selection table locally
# By saving the est object as an RDS file, the data can be easily loaded and used in future R sessions without having to re-run the time-consuming analysis steps, including the resampling step.
save(est,file=est.file.name)



## ----------------- Section Three ---------------------------- 
## Extract sound parameters from .wav files and save them in a dataset for future use

# (1) Measure spectral parameters
# generate spectrograms for each sound recording in the est data frame
sp <- specan(est,parallel=1)
colnames(sp)
# (2) put all features together in a single matrix
# By combining all of these columns into a single data frame, prms, it becomes possible to explore and compare the various spectral and temporal features of the audio recordings using visualization and statistical analysis tools.
# keep the file names of each sound (here this is just column 1)
prms <- data.frame(est[, c("sound.files")], sp[, -c(1:4)])
View(prms)
# View the sound features we got 
colnames(prms)

# (3) save feature measurements so we can just load these later and skip feature extraction
write.csv(prms,file.path(sound_folder ,"acoustic parameters.csv"),row.names = FALSE)
prms <- read.csv(file.path(sound_folder ,"acoustic parameters.csv"), stringsAsFactors = FALSE)
# edit columns names for clarity
old.names <- names(prms)
to.change <-which(old.names %like% "sound.files")
new.names <- old.names
new.names[to.change] <- "sound.files"
colnames(prms) <- new.names 
colnames(prms) # these sound features are extraxted by the default setting of the WarbleR

# refine sound features and get our data set for PCA and cluster analysis
alldata <- select(prms,-"sound.files",-"freq.Q25",-"freq.Q75",-"freq.IQR",-"time.Q25",-"time.Q75",-"time.IQR",-"mindom",-"maxdom",-"dfslope",-"startdom",-"enddom",-"meanpeakf")
colnames(alldata)

## ----------------- Section Four ---------------------------- 
## Create PCA 
alldata.forpca <- alldata
# running pca, omitting NA, and centering
my.pca <- prcomp(na.omit(alldata.forpca, center=T))     
summary(my.pca) # the first 4 components describe 84.47% variance
str(my.pca)

# get standard deviations of the principal components (PCs).
sd <- my.pca$sdev
# get weights or coefficients of each original variable in the principal components
# to identify which original variables are most strongly associated with each principal component.
loadings <- my.pca$rotation
loadings
rownames(loadings) <- colnames(alldata.forpca)
# Get principal component scores
scores <- my.pca$x
# cutoff for 'important' loadings
# calculating the cutoff threshold for principal component scores, below which the score will be considered insignificant. 
loadings
pca.cutoff<- sqrt(1/ncol(alldata.forpca)) 
pca.cutoff # This means that any principal component score with an absolute value less than  0.1428571 can be considered as noise and ignored.

# Visualize PCA: using "factoextra package"
# Extract the eigenvalues/variances of principal components
get_eigenvalue(my.pca)  
# Visualize the eigenvalues
fviz_eig(my.pca)# so keep the first four components
# Visualize the results individuals and variables, respectively.
fviz_pca_ind(my.pca)    
fviz_pca_var(my.pca) 
# Biplot of individuals and variables.
fviz_pca_biplot(my.pca) 
# variable representation in the first 2 PCAs
fviz_cos2(my.pca, choice = "var", axes = 1:4) 


## ----------------- Section Five ---------------------------- 
## Conduct and compare different types of cluster analysis

## (1) CLUSTERING TYPE 1A: K-MEANS CLUSTERING
# removes rows with missing values
alldata.noNA<- na.omit(alldata.forpca)

# Scaling the data
alldata.scaled <- na.omit(scale(alldata.forpca)) 

# determining number of clusters to specify
# visualize the average within-cluster sum of squares (WSS) for different values of k (number of clusters) and to identify the optimal number of clusters based on the "elbow" method.
fviz_nbclust(alldata.scaled, kmeans, method = "wss") + 
  geom_vline(xintercept = 2, linetype = 2)    #using the "elbow" method --> set xintercept = 3

# performs k-means clustering 
kmeans.result <- kmeans(alldata.scaled, 2, nstart = 50)

# visualize clustering results 
fviz_cluster(kmeans.result,alldata.noNA, ellipse.type = "norm")+
  geom_vline(xintercept = 2, linetype = 2) # this method suggests 3 clusters


## (2) CLUSTERING TYPE 1B: PAM CLUSTERING (TYPE OF K-MEANS CLUSTERING)
# automatically determines number of clusters
fviz_nbclust(alldata.scaled, kmeans, method = "silhouette") # this method suggests 2 clusters
pam.result <- pam(alldata.scaled, 2)
print(pam.result)
fviz_cluster(pam.result,alldata.noNA, ellipse.type = "norm")
#this method suggests 2 clusters

## (3) Clustering type 2: Model-based clustering
mc.result<- Mclust(alldata.noNA)
summary(mc.result)
# BIC values used in mclust for choosing the number of clusters
fviz_mclust(mc.result, "BIC", palette = "jco") # this method suggests 2 clusters
# selected "xxx" (ellipsoidal multivar. normal) model with 2 cluster...


## (4) Clustering type 3: Hierarchical Clustering on Principal Component
# (combining PCA and clustering methods)
alldata.noNA<- na.omit(alldata.forpca)

# decided to keep top 2 PCAs based on scree plot from above
pca.result <- PCA(alldata.noNA, ncp = 2, graph = TRUE) 

#Compute and graph the HCPC
hcpc.result <- HCPC(pca.result, graph =TRUE) # cut at 2              

#plotting HCPC clusters
fviz_cluster(hcpc.result,                               
             repel = TRUE,  #Avoid label overlapping
             show.clust.cent = TRUE, #Show cluster centers
             palette = "jco",                              
             ggtheme = theme_minimal(),
             main = "Factor map")

summary(hcpc.result)
hcpc.result$call  # this method suggests 2 clusters




