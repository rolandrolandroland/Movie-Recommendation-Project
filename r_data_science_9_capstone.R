# Capstone Project

###### IMPORTANT README ########
'''
First, the major sections that are commented out are commented out to save time when running
The vectors that they would generate are generated manually below (with all of the save values)

Second, the Rmd file is compiled using render() at the bottom of this file.  You may have to change
The working directory first.  If you just try and knit the rmd file, it will not work because
it requires objects that are stored here
'''

library(knitr)
library(rmarkdown)
library(grid)
library(gridExtra)

#First, set up the environm
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


# I want to further divide my data into a train and test set, so that I can test my data before the final application to 
# the validation set
# I will make the algorithm using the train set edx_train, then calculate the RMSE by applying it to edx_test.  
# Then I will take the algorithm with the lowest RMSE and apply it to the validation set to get my final RMSE
train_index = createDataPartition(y = edx$rating, times = 1, p = .8, list = FALSE)
edx_train = edx[train_index,]
edx_pre_test = edx[-train_index,]

edx_test = edx_pre_test %>% semi_join(edx_train, by = 'movieId') %>% 
  semi_join(edx_train, by = 'userId')
removed = anti_join(edx_pre_test, edx_test)
edx_train = rbind(edx_train, removed)

size_edx_train = object.size(edx_train)
size_edx_train
# define RMSE function
RMSE = function(y, y_hat){
  sqrt(mean((y-y_hat)^2))
}


# just use average of all ratings for each prediction
mu= mean(edx_train$rating) # assign the average rating for all movies to mu
mu # 3.51
RMSE_initial = RMSE(mu, edx_test$rating)  # get RMSE value for this model

RMSE_initial # 1.06.  This is much to high

# Create data frame to store RMSE values from different methods. 
RMSE_results = data_frame(method = "Just the Average", RMSE = RMSE_initial)
RMSE_results


# Model 1: Find the average for each movie and use it as the prediction for each new rating of that movie
# b_i will be will be the average value that each movie is above the average mu
# store it in edx_model_1
edx_model_1 = edx_train %>% group_by(movieId) %>% 
  summarize(b_i = mean(rating) - mu, movie_avg = mean(rating)) %>%
  ungroup()     

# create model_1_predictions with the predicted rating for each movie in the test set edx_test
model_1_predictions = mu+ edx_test %>% left_join(edx_model_1, by = 'movieId') %>% .$b_i

# Store the RMSE for this model in RMSE_model_1 and add it to the RMSE_results data frame
RMSE_model_1 =RMSE(model_1_predictions, edx_test$rating) # = .944

# Bind the results to the results data frame
RMSE_results = bind_rows(RMSE_results, data_frame(method = 'Model 1', RMSE = RMSE_model_1))
RMSE_results


# Model 2: Average for the users
# incorporate the user average in the model
# b_u = b(user) is the average that each user rates above a movie's average rating
edx_train_model_2 = edx_model_1 %>% 
  left_join(edx_train, by = 'movieId') %>% 
  group_by(userId) %>% summarize(b_u = mean(rating - mu - b_i))
head(edx_train_model_2) # userId and user avgs

# make predicted_rating_model_2 with b_u, b_i, and the prediction (mu + b_u + b_i) for each rating in the test set
predicted_rating_model_2 = edx_test %>% left_join(edx_model_1 , by = 'movieId') %>% 
  left_join(edx_train_model_2, by = 'userId') %>% mutate(pred = mu + b_u + b_i)
head(predicted_rating_model_2)
RMSE_model_2 = RMSE(edx_test$rating, predicted_rating_model_2$pred)
RMSE_model_2 # .867
RMSE_results = bind_rows(RMSE_results, data_frame(method = 'Model 2', RMSE = RMSE_model_2))
RMSE_results 



# Model 3: Use regularization to penalize large estimates from small sample sizes
#find best lambda - test all integers 1 through 10
# first, try it on the training set
# I am commenting this out just to make running the code quicker
'''lambdas = seq(0,10,1)
rmses = sapply(lambdas, function(lambda){
  model = edx_train %>% group_by(movieId) %>%
    summarize(b_i = sum(rating-mu)/(n()+lambda), n_i = n(), name = first(title)) %>%
    right_join(edx_train, by = "movieId") %>%
    mutate(pred = mu + b_i) 
  return(RMSE(model$pred, edx_train$rating))
})
lambdas[which.min(rmses)]
rmses[1]# this tells me that a lambda is 0 is optimal for the train_set
# this is because no correction is needed when testing on the same set used for developing the algorithm
'''

# when I use this on the train set, the best lambda is 0, but on the test set it is 2.5.  But the difference is minimal regardless
lambdas_model_3 <- seq(0, 10, 0.25)
'''  Comment it out to save time when running
RMSES_model_3 = sapply(lambdas_model_3, function(l){
  model = edx_train %>% group_by(movieId) %>%
    summarize(b_i = sum(rating-mu)/(n()+l), n_i = n(), name = first(title)) %>%
    right_join(edx_test, by = "movieId") %>%
    mutate(pred= mu + b_i)
  return(RMSE(model$pred, edx_test$rating))
})
'''
#Results are manually stored in vector below
RMSES_model_3 = c(0.9443640, 0.9443426, 0.9443276, 0.9443165, 0.9443082, 0.9443020, 0.9442975, 0.9442943, 0.9442923,
                  0.9442912, 0.9442909, 0.9442914, 0.9442926, 0.9442944, 0.9442967, 0.9442995, 0.9443028, 0.9443065,
                  0.9443106, 0.9443150, 0.9443198, 0.9443248, 0.9443302, 0.9443358, 0.9443417, 0.9443479, 0.9443542,
                  0.9443608, 0.9443676, 0.9443746, 0.9443818, 0.9443891, 0.9443966, 0.9444042, 0.9444120, 0.9444200,
                  0.9444281, 0.9444363, 0.9444446, 0.9444531, 0.9444616)
# make a plot of RMSE vs Lambda for this model
RMSE_plot_model_3 = data.frame(lambdas_model_3, RMSES_model_3) %>%
  ggplot(aes(lambdas_model_3, RMSES_model_3)) + geom_point() + xlab("Lambda") + ylab("RMSE")+
  ggtitle("RMSE vs Lambda in Model 3") +
  theme(plot.title = element_text(hjust = 0.5, size = 10))
  
RMSE_plot_model_3

lambdas_model_3[which.min(RMSES_model_3)]
# a lambda of 2.5 minimizes the RMSE to a value of 0.944
RMSE_model_3 = min(RMSES_model_3)
RMSE_results = bind_rows(RMSE_results, data_frame(method = 'Model 3', RMSE = RMSE_model_3))
RMSE_results

# Model 4: regularization on the user effect
lambdas_model_4 = seq(0,10,.25)
'''
RMSES_model_4_a = sapply(lambdas_model_4, function(l){
  b_i = edx_train %>% group_by(movieId) %>%
    summarise(b_i = sum(rating-mu)/(n()+2.5)) # include regularized movie effect
  b_u = edx_train %>% left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu- b_i)/(n()+l))
  predicted_rating = edx_test %>% left_join(b_i, by ="movieId") %>% left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) 
  return(RMSE(predicted_rating$pred, edx_test$rating))
})
RMSES_model_4_a = paste(RMSES_model_4_a, collapse = ", ")
''' # Again manually create vectors to save time when running
RMSES_model_4_a = c(0.866520846936756, 0.866456474719912, 0.866397458613238, 0.866343466548302, 0.86629419262024, 0.866249354536074,
                    0.866208691360753, 0.866171961520435, 0.866138941028806, 0.86610942190743, 0.866083210775437, 0.866060127587425,
                    0.866040004501484, 0.866022684861731, 0.866008022281923, 0.865995879818461, 0.865986129222691, 0.865978650263647, 
                    0.865973330113564, 0.865970062789396, 0.865968748644424, 0.865969293904744, 0.865971610246048, 0.865975614406633, 
                    0.865981227833045, 0.865988376355173, 0.865996989887956, 0.866007002157177, 0.866018350447094, 0.866030975367896, 
                    0.866044820641175, 0.866059832901809, 0.866075961514786, 0.86609315840569, 0.866111377903644, 0.866130576595667, 
                    0.866150713191483, 0.866171748397912, 0.866193644802057, 0.866216366762576, 0.866239880308392)
lambdas_model_4[which.min(RMSES_model_4_a)] # a lambda of 5
min(RMSES_model_4_a) # .8659687
# Make a plot of RMSE vs Lambda for this model
RMSE_plot_model_4_a = data.frame(lambdas_model_4, RMSES_model_4_a)%>%
  ggplot(aes(lambdas_model_4,RMSES_model_4_a)) +
  geom_point() + xlab("Lambda") + ylab("RMSE") + 
  ggtitle("RMSE vs Lambda in Model 4a") +
  theme(plot.title = element_text(hjust = 0.5, size = 10))
RMSE_plot_model_4_a


# Now I rerun it, but this time the lambda(b_u) will be fixed at 5 and lambda(b_i) will be optimized
'''
RMSES_model_4_b = sapply(lambdas_model_4, function(l){
  b_i = edx_train %>% group_by(movieId) %>%
    summarise(b_i = sum(rating-mu)/(n()+l)) # include regularized movie effect
  b_u = edx_train %>% left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu- b_i)/(n()+5))
  predicted_rating = edx_test %>% left_join(b_i, by ="movieId") %>% left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) 
  return(RMSE(predicted_rating$pred, edx_test$rating))
})

RMSES_model_4_b = paste(RMSES_model_4_b, collapse = ",")
'''  # Manually create vector so that funcion does not have to run every time I start RStudio
RMSES_model_4_b = c(0.866125464265412,0.86609262617991,0.866067349638214,0.866046882789005,0.866029837528324,0.86601540550868,
                    0.86600306710353,0.865992463670843,0.86598333419601,0.865975480814814,0.865968748644424,0.865963013281571,
                    0.865958172678194,0.865954141650256,0.865950848041234,0.865948229964694,0.865946233773381,0.865944812531205,
                    0.865943924841875,0.865943533935952,0.865943606948769,0.865944114341733,0.865945029433027,0.865946328012935,
                    0.86594798802548,0.865949989302588,0.865952313340347,0.865954943109288,0.865957862892427,0.865961058146147,
                    0.865964515380005,0.865968222052336,0.865972166479131,0.865976337754135,0.865980725678474,0.865985320698445,
                    0.865990113850296,0.865995096711043,0.866000261354534,0.866005600312054,0.86601110653692)

min(RMSES_model_4_b) # .8659435
lambdas_model_4[which.min(RMSES_model_4_b)] # lambda(b_i) = 4.75 gives the ideal value
RMSE_model_4=min(RMSES_model_4_b)
RMSE_results = bind_rows(RMSE_results, data_frame(method = "Model 4", RMSE = RMSE_model_4))
RMSE_results
# Make a plot of RMSE vs Lambda for this model
RMSE_plot_model_4_b = data.frame(lambdas_model_4, RMSES_model_4_b) %>% 
  ggplot(aes(lambdas_model_4, RMSES_model_4_b)) + 
  geom_point() + xlab("Lambda") + ylab("RMSE") +
  ggtitle("RMSE vs Lambda in Model 4b")+
  theme(plot.title = element_text(hjust = 0.5, size = 10))
RMSE_plot_model_4_b

# create regularized b_i and b_u for use later
b_i = edx_train %>% group_by(movieId) %>%
  summarise(b_i = sum(rating-mu)/(n()+ 4.75)) 
b_u = edx_train %>% left_join(b_i, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu- b_i)/(n()+5))



# Model 5: The Genre Effect
# Create g_u, which contains the average of the differences between the expectedrating that a user gave a movie (mu + b_i + b_u) 
# and the average rating that that user gave movies in that genre
# I am defining genre as combination of all genres assigned to a movie.  Each movie has 1 genre.
g_u = edx_train %>% left_join(b_u, by = 'userId') %>% left_join(b_i, by = 'movieId') %>% group_by(userId, genres) %>%
  summarize(n = n(), g_u = mean(rating -mu - b_u -b_i))

# make one data frame iwht g_u, b_u, b_i, a pred, and the other info
g_u_pred = edx_train %>% left_join(b_u, by = 'userId')%>% left_join(b_i, by = 'movieId') %>% left_join(g_u, by = c('userId', 'genres'))%>%
  mutate(pred = mu + b_u + b_i + g_u, res = rating - pred) 

RMSE_Model_5_a = RMSE(g_u_pred$pred, g_u_pred$rating) # 0.5665284
#only so low because of overtraining.  There are many movies that are the only movie of that particular genre that a user rated

# test on test set
g_u_test = edx_test %>% left_join(b_u, by = 'userId') %>% left_join(b_i, by = 'movieId') %>%
  left_join(g_u, by = c('userId', 'genres')) 

# There will be many genres that are new to a user.  Just assign g_u to 0 for all those
g_u_test$g_u[is.na(g_u_test$g_u)] = 0 
g_u_test  = g_u_test%>% mutate(pred = mu + b_u + b_i +g_u)
RMSE(g_u_test$pred, edx_test$rating) # 0.9218755
#the RMSE went up, but I bet that I can decrease it if I punish g_u values with low samples.

# use sapply to find regularize the effect
lambda_model_5_b = seq(1, 10, 1)
"""
RMSES_model_5_b = sapply(lambda_model_5_b, function(l){
  g_u_func = edx_train %>% left_join(b_u, by = 'userId') %>% left_join(b_i, by = 'movieId') %>%
    group_by(userId, genres) %>%
    summarize(g_u = sum(rating-mu-b_i-b_u)/(n()+l))
  #head(g_u_func)
  g_u_pred_func = edx_test %>% left_join(b_u, by = 'userId')%>% left_join(b_i, by = 'movieId') %>% 
    left_join(g_u_func, by = c('userId', 'genres'))
  g_u_pred_func$g_u[is.na(g_u_pred_func$g_u)] = 0
  g_u_pred_func= g_u_pred_func %>% mutate(pred = mu + b_u + b_i + g_u) 
  return((RMSE(g_u_pred_func$pred, edx_test$rating)))
})
RMSES_model_5_b = paste(RMSES_model_5_b, collapse = ',')
""" # Manually Create vector again
RMSES_model_5_b = c(0.87525711950922,0.864966610523731,0.861129960849642,0.859389302757039,0.858534747811657,
                    0.85811427667515,0.85792688479025,0.85787232101276,0.857896372548049,0.857967927353733)

lambda_model_5_b[which.min(RMSES_model_5_b)]
min(RMSES_model_5_b) # .8578723 when lambda = 8

RMSE_plot_5_b = data.frame(lambda_model_5_b, RMSES_model_5_b) %>%
  ggplot(aes(lambda_model_5_b, RMSES_model_5_b)) + geom_point() +
  xlab("Lambda") + ylab("RMSE")+
  ggtitle("RMSE vs Lambda in Model 5b")+
  theme(plot.title = element_text(hjust = 0.5, size = 10))
RMSE_plot_5_b

# run again to narrow it down
lambda_model_5_c = seq(7, 9, 0.1)
"""
RMSES_model_5_c = sapply(lambda_model_5_c, function(l){
  g_u_func = edx_train %>% left_join(b_u, by = 'userId') %>% left_join(b_i, by = 'movieId') %>%
    group_by(userId, genres) %>%
    summarize(g_u = sum(rating-mu-b_i-b_u)/(n()+l))
  #head(g_u_func)
  g_u_pred_func = edx_test %>% left_join(b_u, by = 'userId')%>% left_join(b_i, by = 'movieId') %>% 
    left_join(g_u_func, by = c('userId', 'genres'))
  g_u_pred_func$g_u[is.na(g_u_pred_func$g_u)] = 0
  g_u_pred_func= g_u_pred_func %>% mutate(pred = mu + b_u + b_i + g_u) 
  return((RMSE(g_u_pred_func$pred, edx_test$rating)))
})
RMSES_model_5_c = paste(RMSES_model_5_c, collapse = ",")
""" # Manually Create vector again
RMSES_model_5_c = c(0.85792688479025,0.857916601230305,0.857907543656917,0.857899648719994,0.857892856695401,
                    0.857887111244101,0.857882359189722,0.857878550312934,0.857875637161211,0.857873574872662,
                    0.85787232101276,0.85787183542289,0.857872080079754,0.857873018964748,0.857874617942504,
                    0.857876844647881,0.857879668380729,0.857883060007819,0.857886991871402,0.857891437703865,
                    0.857896372548049)

lambda_model_5_c[which.min(RMSES_model_5_c)] # 0.8578718 when lambda = 8.1
RMSE_model_5 =min(RMSES_model_5_c)
RMSE_model_5
RMSE_results = bind_rows(RMSE_results, data_frame(method = "Model 5", RMSE = RMSE_model_5))
RMSE_results


#RMSE of validation set
g_u_valid = edx_train %>% left_join(b_u, by = 'userId') %>% left_join(b_i, by = 'movieId') %>%
  group_by(userId, genres) %>%
  summarize(g_u = sum(rating - mu - b_i-b_u)/(n()+8.1))
g_u_pred_valid = validation %>% left_join(b_u, by = 'userId') %>% left_join(b_i, by = 'movieId') %>%
  left_join(g_u_valid, by = c('userId', 'genres'))
g_u_pred_valid$g_u[is.na(g_u_pred_valid$g_u)] = 0
g_u_pred_valid = g_u_pred_valid %>% mutate(pred = mu + b_u + b_i + g_u, res = abs(pred - rating))
RMSE_final = RMSE(g_u_pred_valid$pred, validation$rating) # 0.8577153
RMSE_results = bind_rows(RMSE_results, data_frame(method = "Final RMSE", RMSE = RMSE_final))
RMSE_results
sum(g_u_pred_valid$g_u==0)/nrow(g_u_pred_valid) # This correction is only being applied to 36.572 % of the data
RMSE_final-.8649

# Analysis of residuals in validation set predictions
# First, plot the mean residual vs number of ratings
''' This plot is nice, but too large to include in the report
g_u_valid_plot = g_u_pred_valid %>% group_by(userId, genres) %>% 
  summarize(n = n(),mean_g_u = mean(g_u), mean_res = mean(res))
  ggplot(aes(n, mean_res)) + geom_point() + xlab("Number of Ratings") + ylab("Mean Residual") +
  ggtitle("Mean Residual vs Number of Ratings Defining User-Genre Groups")+
  theme(plot.title = element_text(hjust = 0.5, size = 10))
g_u_valid_plot
'''



# Now I will plot the mean residual as a function of number of ratings in the genre-user group.
# 'n' is the number of ratings in that user-genre group
g_u_pred_valid_1_plot=g_u_pred_valid %>% group_by(userId, genres) %>% 
  summarize(n = n(),mean_g_u = mean(g_u), mean_res = mean(res)) %>%
  group_by(n) %>% summarize(num = n(), res_mean = mean(mean_res)) %>% 
  filter(n<=30) %>%
  ggplot(aes(n, res_mean)) + geom_point() + geom_smooth(method='lm')+
  xlab("Number of Ratings") + 
  ylab("Mean Residual") +
  ggtitle("Mean Residual vs Number of Ratings Defining User-Genre Groups")+
  theme(plot.title = element_text(hjust = 0.5, size = 10))
g_u_pred_valid_1_plot

g_u_pred_valid_1_table = g_u_pred_valid %>% group_by(userId, genres) %>%
  summarize(n = n(), mean_res = mean(res)) %>%
  group_by(n) %>% summarize(num = n(), res_mean = mean(mean_res)) %>% 
  top_n(-15, n) %>% arrange(n)
colnames(g_u_pred_valid_1_table) =c('Number of Ratings Defining User-Genre Group', 'Size of Group', 'Mean Residual')
g_u_pred_valid_1_table


### PLOTS
lambda_plots = grid.arrange(RMSE_plot_model_3, RMSE_plot_model_4_a, RMSE_plot_model_4_b, RMSE_plot_5_b, ncol = 2)

## Render rmd file
list.files()
setwd("Desktop/R_Data_Science_Course")
render("r_data_science_9_capstone_writeup.Rmd", output_format = "pdf_document")


