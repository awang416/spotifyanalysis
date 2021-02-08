"""
-duration_ms   int  the duration of the track in milliseconds
-key  int  the estimated overall key of the track. Integers map to pitches using Standard Pitch Class notation
-mode int  	Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived.
Major is represented by 1 and minor is 0.
-acousticness float 	 A confidence measure from 0.0 to 1.0 of whether the track is acoustic.
1.0 represents high confidence the track is acoustic.
-danceability float   Danceability describes how suitable a track is for dancing based on a combination of musical elements
including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
-energy  float  	Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity.
Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy,
Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
-instrumentalness float, Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context.
Rap or spoken word tracks are clearly “vocal”. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0
-liveness float   	Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live.
A value above 0.8 provides strong likelihood that the track is live.
-loudness float   	The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks.
-speechiness float  Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording, the closer to 1.0 the attribute value.
Values above 0.66 describe tracks that are probably made entirely of spoken words.
-valence  float     A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric).
-tempo    float   The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
-popularity integer  	The popularity of the track. The value will be between 0 and 100, with 100 being the most popular.
The popularity of a track is a value between 0 and 100, with 100 being the most popular.

"""

import argparse
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
import time


def extract_features (df, min_threshold):
    """

    :param df:  the input of this method will be a panda data frame.
    Specifically, in this data frame, the genre can be separated into more
    different features, such as country type and music genre. Each row has
    different genre, and that's too much of diversity, we want to extract out
    some similarity between the genres
    :return: a panda dataframe that separates the features contained in
    genre column, the idea for separate here is every genre data here consists of
    an adjective and a noun. The original data here contains something like "American Pop",
    which narrows down the big area of Pop into even smaller field. This dataset is designed
    intentionally to make every row have different genre, and that's too much diversity for
    our project, so we want to extract some similarity between rows.

    Operations:
    1. We will extract the adj as adjective_genres; some examples involve "abstract" in the
    original genres such as "abstract pop" and "abstract rap". On the other hand, there will be
    the noun part in the genre to be extracted.
    2. Moreover, there are some cases that the genre only includes 1 word, for instance, the first
    row of the original dataset has genre value of "432hz", which is not related to other genres.
    However, some other rows, such as the genre of row 4, have genre value to be the same as the adjective
    of other rows; row 4's genre is "abstract", row 5's genre is "abstract beats". Thus for the goal of looking
    for some common patterns in the genre pattern to help predict popularity, we save those rows that have
    single genre value that acts as an adjective for other rows, and we will drop all the other rows that
    have nothing to do with other rows; those rows are too diverse.
    3. Convert each adj and noun value left to numeric value. There will be 756 types of adjectives and
    747 types of nouns. We give each of them a numeric value to represent.
    """

    print("NOW ENTER THE FEATURE EXTRACTION STAGE!")
    print("THIS PART MAINLY DEALS WITH THE \"GENRE\" COLUMN")
    print()
    print("1. SPLIT GENRES INTO ADJECTIVE LIST & NOUN LIST, SINGLE WORD WILL BE INSERTED IN BOTH LISTS INITIALLY ")
    """
    Take out genre column and set up some initial lists. Adjective list to store adjectives in the genre
    and noun list to store all the rest of information in genre, which is generally a noun. It might not
    necessarily be one word, for example, it can be "hip hop"
    """
    genre_column = df['genres']
    list_specific_adjective = []
    list_specific_noun = []

    """
    Iterate over each row in the "genre" column. Take out the index of first empty space to 
    split adjectives and nouns. If there is no empty space, assign the same genre value to
    both adjective and noun. Also if the adjective word is a single-letter word, such as "a" or "n",
    it is more likely that the adjective is also part of the noun, so that is one more condition to
    consider
    """
    for index, value in genre_column.items():
        first_empty_space = value.find(' ')
        # Test whether there is no word after the first empty space
        if first_empty_space != -1 and first_empty_space != 1:
            the_specific_adjective = value[:first_empty_space]
            list_specific_adjective.append(the_specific_adjective)
            list_specific_noun.append(value[first_empty_space+1:])
        else:
            list_specific_adjective.append(value)
            list_specific_noun.append(value)
    print("The adjective list looks like:", end= " ")
    print(list_specific_adjective)
    print("The noun list looks like:", end= " ")
    print(list_specific_noun)
    print()


    """
    Work on dropping "useless" rows. Only single word genres can possibly be a useless row.
    It will be useless if it's not related to any other row at all. We use number of frequency
    to test. If the frequency is 1, which means this genre is pretty unique. Then containing this
    genre value will be meaningless, since it contains so much unique genres. 
    Thus, we want to drop anything that's too diverse or unique. We prefer more of
    a general pattern for prediction. Also we will drop the entire row in the panda dataframe to
    maintain the same dimensionality. We will check the occurring frequency of 
    each single word in adjective and noun lists
    """

    count_useless = 0
    list_row_drop = []
    list_multiple_in_adj = [] # the list of single word that acts as an adjective for other words, e.g. alternative, since there is alternative rock, alternative hip hop
    list_multiple_in_noun = [] # the list of single word that acts as a noun for other words, e.g. baroque, since there is British baroque
    list_unique = [] # those that really just appear once
    count_single_word = 0 # a counter for recording how many single words appear.
    for index in range(len(list_specific_adjective)):
        if list_specific_noun[index] == list_specific_adjective[index]:
            count_single_word += 1
            # The number of occurences of a single word as an adjective or a noun
            frequency_in_adj = list_specific_adjective.count(list_specific_adjective[index])
            frequency_in_noun = list_specific_noun.count(list_specific_adjective[index])
            if frequency_in_adj > 1:
                list_multiple_in_adj.append(list_specific_adjective[index])
            if frequency_in_noun > 1:
                list_multiple_in_noun.append(list_specific_adjective[index])
            if frequency_in_adj == 1 and frequency_in_noun == 1:
                count_useless += 1
                list_row_drop.append(index)  # drop the row with index that is neither an adjective nor a noun
                list_unique.append(list_specific_adjective[index])
    print("The single word that is an adj: (length: %d)" % len(list_multiple_in_adj), end="")
    print(list_multiple_in_adj)
    print("The single word that is a noun: (length: %d) " % len(list_multiple_in_noun), end="")
    print(list_multiple_in_noun)
    print("The single word that is too unique: (length: %d)" % len(list_unique), end="")
    print(list_unique)
    print()

    print("Check if the total number of elements are the same after excluding overlapping genres")
    print("The length of the union of the three separate lists are: %d" % len(set(list_multiple_in_noun+list_multiple_in_adj+list_unique)))
    print("The length of total single words are: %d" % count_single_word)
    print()

    print("2. EXTRACT THE INDEX OF THOSE USELESS ROWS AND DROP THOSE ROWS FROM THE DATASET")
    print("The total number of useless genres is %d" % count_useless)
    print("The index of rows that contain useless genres:", end=" ")
    print(list_row_drop)
    print()

    for ele in sorted(list_row_drop, reverse=True):
        del list_specific_adjective[ele]
        del list_specific_noun[ele]
    df = df.drop(df.index[list_row_drop]).reset_index(drop=True)
    print("The dataset after dropping rows and resetting index looks like: ")
    print(df)
    print()

    """
    Drop the genres column and add the adj and noun list as two separate columns into 
    the dataframe, we should encode those two columns into numeric values, since 
    simply splitting them doesn't change the datatype of "string". Inspired by a Kaggle 
    project, we intend to apply Target encoding approach, since we are really sure that 
    there should be a correlation between genres and popularity. Though it's really hard
    to tell which genre is the best, we are sure some genres clearly are more likely to be
    prefered by mainstream, such as pop and hiphop over soul. This preference is just 
    similar to the case that some people prefer some artist than the others. You cannot say
    who's the most popular one, but there are clearly some names that immediately pop into 
    your mind. 
    
    Thus, we will replace the two new genre columns with some derivative of its popularity. 
    Our ultimate goal is to have only 1 new column that is a combination of information 
    from adjective genre, noun genre, and the popularity. 
    
    So our plan is as follow: first group by every unique values of adj_nouns and take 
    their mean. For example, there can be Korean pop, Korean hip hop, Korean country,
    so we will take the mean of all rows that have "Korean" and call it adj_average. 
    Similarly we will take all the average of the noun_genre. Then we will add the 
    two mean and divide by 2, and this new value will replace the two genres in a 
    good numerical form. 
    """
    print("3. DROP THE GENRES COLUMN AND ADDING INTO OUR ADJ AND NOUN COLUMNS ")
    df = df.drop(columns=['genres'])
    df['adj_genres'] = list_specific_adjective
    df['noun_genres'] = list_specific_noun
    print(df)
    print()
    print("4. CHECK WHAT ARE SOME GENRES THAT LEAD TO HIGH POPULARITY")
    print("SHOWN IN TWO GRAPHS")
    print()
    top_genres(df, 20, 'adj_genres')
    top_genres(df, 20, 'noun_genres')

    print("5. CHECK THE OVERALL NUMBER OF OCCURRENCE OF ADJ_GENRE & NOUN_GENRE ")
    print("SHOWN IN TWO GRAPHS")
    adj_count = count_appearance(df, 'adj_genres')
    noun_count = count_appearance(df, 'noun_genres')
    print("Conclusion: most of the adj_genres appear less than 35")
    print("Conclusion: most of the noun_genres appear less than 75")
    print()

    print("6. CREATE THE LIST OF NUMERICAL VALUE OF GENRES")
    adj_mean = df.groupby('adj_genres')['popularity'].transform('mean')
    noun_mean = df.groupby('noun_genres')['popularity'].transform('mean')

    popularity_column = df['popularity'].to_frame()
    print("The adjective column after the mean transformation looks briefly like: ")
    print(adj_mean.head())
    print()
    print("The noun column after the mean transformation looks briefly like: ")
    print(noun_mean.head())
    print()

    print("We can also fulfill the same task of applying the agg function,\n"
          "which has the advantage of storing the count, so we know the number\n"
          "of occurrence of each unique adjective genre and each noun genre\n"
          "and it also excludes duplicates as well"
          )
    print()
    adj_mean_count_form = popularity_column.groupby(df['adj_genres']).agg(['mean', 'count'])
    noun_mean_count_form = popularity_column.groupby(df['noun_genres']).agg(['mean', 'count'])
    print("The number of occurrence of each adjective genres: ")
    print(adj_mean_count_form[('popularity', 'count')])
    print()
    print("The number of occurrence of each noun genres: ")
    print(noun_mean_count_form[('popularity', 'count')])
    print()

    print("Also check the max occurrence, we are not interested in min, which is supposed to be 1")
    print("The max number of occurrence in adj genre is %d and it's %d in noun genre" %
          (adj_mean_count_form[('popularity', 'count')].max(), noun_mean_count_form[('popularity', 'count')].max()))
    print()

    print("7. INCORPORATE THE MEAN POPULARITY INFORMATION FROM BOTH ADJECTIVE AND NOUN ")
    print()
    print("Remember in this part, there is the consideration that mean popularity might overfit\n"
          "the real popularity level when the number of occurrence is too small. In that case\n"
          "we replace the rows with too little occurrence with the mean of the whole popularity column ")
    print()
    print("After taking the average of the adj mean popularity and noun mean popularity")
    pop_mean = popularity_column.mean().values[0]

    final_list = []
    for index in range(len(adj_mean)):
        # stands for adjective mean value and noun mean value
        amv, nmv = df.iloc[index, [-2,-1]]
        # stands for Adjective count, Ncount for Noun count
        Acount = adj_mean_count_form.loc[amv, ('popularity', 'count')]
        Ncount = noun_mean_count_form.loc[nmv, ('popularity', 'count')]
        adj_val = adj_mean[index]
        noun_val = noun_mean[index]
        if Acount< min_threshold:
            adj_val = pop_mean
        if Ncount< min_threshold:
            noun_val = pop_mean
        average = (adj_val + noun_val)/2
        final_list.append(average)
    print(final_list)
    print()

    print("8. FINALLY UPDATING THE DATASET. NOW THE NEW DATASET LOOKS LIKE THIS:")
    df = df.drop(columns=['adj_genres', 'noun_genres'])
    df['new_genres'] = final_list
    return df


def feature_selction (df, threshold_popularity, threshold_corr_between, option):
    """
    :param df: The parameter will just be the dataframe after splitting the genre column
    :return: We are now working on selecting the features for linear regression model prediction.
    Some features might be correlated with each other, which means there is a higher chance of redundancy
    that we should deal with. Also, we prefer having more features that have correlation with the
    target variable: popularity. There are lots different ways to select features, for example we may have
    different amount of features selected, or we may have different opinions on paying more attention to
    removing redundancy or maintaining high correlation with popularity column.
    """

    """
    Show heatmap of different features, here we should exclude the column of new_genres, 
    since the design of this new column is based on its correlation with popularity. However,
    we would include it in the heatmap anyway 
    """
    matrix = df.corr(method = 'pearson')
    correlation = np.abs(df.corr())
    fig, ax = plt.subplots(figsize=(13, 13))
    color_map = sb.color_palette("mako")
    sb.heatmap(correlation, cmap=color_map, square=True)
    plt.title('Correlation between features: abs values adjusted')
    plt.show()

    if option == 'option1':
        print("--We enter option1, option1 means we will not do feature selection at all, ")
        print("and we will simply output the dataset after feature extraction stage into our model")
        print()
        return df.drop('popularity', axis=1)
    else:
        print("--We enter %s, which means we will select features based on pearson correlation with popularity " % option)
        print("We passed into a parameter called \"threshold popularity\" into this method, so we will only keep\n"
              "the features that have pearson correlation with popularity that is above that threshold value")
        if option == 'option2':
            print("Remember in this option, we do not consider about removing redundant features, so two features\n"
                  "might have strong correlation")
            print()
        else:
            print("We will also deal with redundant features that have strong correlation within themselves ")
            print()

        """
        Figure out based on the value of Pearson correlation, what are the features that have 
        a correlation with "popularity" that is greater than the threshold value  
        """
        print("The sorted linear correlatin value with popularity for all features are as follow")

        correlation_with_popularity = np.abs(correlation['popularity'])
        sort_correlation_with_popularity = correlation_with_popularity.sort_values(ascending=False)
        print(sort_correlation_with_popularity)
        print()
        print("Apply the threshold value of %.2f, the most linear correlated features to POPULARITY are:" % threshold_popularity)

        feature_strong_corr_popularity = []
        for i, row in enumerate(sort_correlation_with_popularity):
            if row == 1:
                continue
            elif row >= threshold_popularity:
                column_name = correlation_with_popularity[correlation_with_popularity == row].index[0]
                print("{%s} --> {%f} (absolute value)" % (sort_correlation_with_popularity.index[i], row))
                feature_strong_corr_popularity.append(column_name)
            else:
                break

        print()
        df_strong_corr = df[feature_strong_corr_popularity]
        print("The dataset that only selects those strong features looks like:")
        print(df_strong_corr)
        print()

        """
        Also, try showing pairplot. Having more visualization is easier for audience to understand 
        We initially try draw the pair plot for the input dataset in this method, but as there are so many of features
        all the numbers and names of features cluster together, so we'll apply the technique here, we'll try to use the pair 
        plot as a visualization helper 
        """
        print("Check the Pearson Correlation within the features that have strong correlation with \"popularity\" ")
        print("The pairplot is shown in a plot")
        print()
        pair_plot = sb.pairplot(df_strong_corr.sample(20), height=1.5,
                                vars=df_strong_corr.columns.values)
        plt.show()

        if option == 'option3':

            """
            Now working on removing redundancy, check if there is any feature that is correlated with others
            """
            matrix2 = df_strong_corr.corr(method='pearson')
            print("The correlation matrix between the 7 features (that are most correlated with \"popularity\") looks like: ")
            print(matrix2)

            redundancy_threshold = threshold_corr_between
            col = matrix2.columns.values
            corr_list = {}

            # for each features, list the other features that have correlation higher than threshold
            for i in range(len(col)):
                elem = list()
                for j in range(len(col)):
                    if i != j:
                        if abs(matrix2.iloc[i, j]) >= redundancy_threshold:
                            elem.append(col[j])
                if elem:
                    corr_list['' + col[i]] = elem

            print()
            print('List of features with correlations higher than threshold of ' + str(redundancy_threshold))
            print(corr_list)
            print()

            # From correlation list, energy was used and acousticness and loudness are removed for redundancy
            df_strong_corr = df_strong_corr.drop(columns=['acousticness', 'loudness'])

        print("-"*100)
        return df_strong_corr



def lr_train(xTrain, yTrain):
    lr_model = LinearRegression().fit(xTrain, yTrain)
    return lr_model

def knn_train(xTrain, yTrain, n):
    yTrain = np.where(yTrain >= np.mean(yTrain), 1, 0)
    knn_model = KNeighborsClassifier(n_neighbors=n).fit(xTrain, yTrain.astype('int'))
    return knn_model

def dt_train(xTrain, yTrain, n, m):
    yTrain = np.where(yTrain >= np.mean(yTrain), 1, 0)
    dt_model = DecisionTreeClassifier(max_depth = m, min_samples_leaf = n).fit(xTrain, yTrain.astype('int'))
    return dt_model

def lr_predict(model, xTest):
    yHat = model.predict(xTest)
    return yHat

def knn_predict(model, xTest):
    yHat = model.predict(xTest)
    return yHat

def dt_predict(model, xTest):
    yHat = model.predict(xTest)
    return yHat


"""
THIS SECTION INVOLVES SOME HELPER METHOD
"""

def scaling(df):
    # use the standard scaler here
    scaler = MinMaxScaler()
    scaler.fit(df)
    df_after_normalize = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return df_after_normalize

def extract_target(df):
    y_data = pd.DataFrame()
    y_data['popularity'] = df['popularity']
    return y_data

def ds_introduction(df):
    print(df.head())
    print(df.info())
    print()
    numerical_columns = df.columns[df.dtypes != 'object']
    string_columns = df.columns[df.dtypes == 'object']
    print("There are %d numeric columns & %d string columns"
          % (len(numerical_columns),len(string_columns)))
    print("-" * 100)
    print("-" * 100)

    # Check if there is any null value or duplicated value
    print("CHECK NUMBER OF DUPLICATES & NULLS")
    print()
    print("The number of duplicates are: %d" % df.duplicated().sum())
    print("The number of nulls are: %d" % df.isnull().sum().sum())
    print("-" * 100)
    print("-" * 100)

    # The main problem comes from the one "String column"---the genre feature
    print("WE REALIZE THAT EACH ROW HAS A DIFFERENT GENRE")
    print("THAT'S TOO MUCH DIVERSITY, SOME ROWS JUST HAVE SUBTLE DIFFERENCES. WE WANT TO EXTRACT "
          "SOME SIMILARITY")
    print()
    print("the dimensionality of dataset:", end=" ")
    print(df.shape)
    print("genres: %d" % df['genres'].nunique() + " unique values")
    print("-" * 100)
    print("-" * 100)

def top_genres(df, num, feature):
    fig, ax = plt.subplots(figsize=(12, 10))
    top_genres = df.groupby(feature)['popularity'].sum().sort_values(ascending=False).head(num)
    ax = sb.barplot(x=top_genres.values, y=top_genres.index, palette="Greens", orient="h", edgecolor='black',
                     ax=ax)
    ax.set_xlabel('Sum of Popularity', c='r', fontsize=12)
    ax.set_ylabel(feature, c='r', fontsize=12)
    ax.set_title('%d Most Popular %s in Dataset' % (num,feature), c='r', fontsize=14, weight='bold')
    plt.show()

def count_appearance(df, feature):
    count = df.groupby(feature)['popularity'].transform('count')
    # plotting
    fig, ax = plt.subplots(figsize=(15, 5))
    ax = sb.distplot(count, bins=600)
    ax.set_xlabel('Count of apperances in data', fontsize=12, c='r')
    ax.set_ylabel('percentage of %s' % feature, fontsize=12, c='r')
    plt.show()
    return count


def split(original_df, select_df):
    y_list = original_df['popularity']
    myTrain, myTest, trainY, testY = train_test_split(select_df, y_list, test_size=0.2, random_state=42)
    return myTrain, myTest, trainY, testY


def plot_lr(y_predict, y_real):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax = sb.scatterplot(x=y_real, y=y_predict)
    sb.lineplot(x=y_real, y=y_real, color='red', ax=ax)
    ax.set_xlabel('Y_test')
    ax.set_ylabel('Y_test_pred')
    ax.set_title('y_test vs. y_test_pred', fontsize=14, color='red')
    plt.show()

def plot_knn(y1, y2, y3):
    x = range(2, len(y1) + 2)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax = sb.lineplot(x=x, y=y1, color='red', ax=ax)
    sb.lineplot(x=x, y=y2, color='green', ax=ax)
    sb.lineplot(x=x, y=y3, color='blue', ax=ax)
    ax.set_xlabel('Neighbors')
    ax.set_ylabel('RMSE')
    ax.set_title('Finding optimal n-neighbors', fontsize=14, color='red')
    plt.show()

def plot_dt(y1, y2, y3):
    x = range(2, len(y1) + 2)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax = sb.lineplot(x=x, y=y1, color='red', ax=ax)
    sb.lineplot(x=x, y=y2, color='green', ax=ax)
    sb.lineplot(x=x, y=y3, color='blue', ax=ax)
    ax.set_xlabel('Minimum Leaf Sample')
    ax.set_ylabel('RMSE')
    ax.set_title('Finding optimal decision tree', fontsize=14, color='red')
    plt.show()


def lr_bagging(xFeat, y, num):
    xFeat_numpy = xFeat.to_numpy()
    y_numpy = y.to_numpy()
    y_predict_list = []
    info_list = []
    for i in range(0, num):
        dictionary = {}
        x_row = np.random.choice(xFeat_numpy.shape[0], replace=True, size=xFeat_numpy.shape[0])
        bootstrap_x = xFeat_numpy[x_row,:]
        bootstrap_y = y_numpy[x_row]
        linear_model = lr_train(bootstrap_x, bootstrap_y)
        dictionary['lr'] = linear_model
        dictionary['train_row_index'] = x_row
        info_list.append(dictionary)
    for row_index in range(len(xFeat_numpy)):
        y_sum = 0
        count = 0
        for tree_index in range(len(info_list)):
            if row_index not in info_list[tree_index]['train_row_index']:
                count += 1
                row = xFeat_numpy[row_index, :]
                row = row.reshape(1, -1)
                y_sum = y_sum + info_list[tree_index]['lr'].predict(row)
        y_average = y_sum/count
        y_predict_list.append(y_average[0])


    return y_predict_list


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line & load the input data file
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",
                        default="data_by_genres.csv",
                        help="filename of our input data")
    args = parser.parse_args()
    data_file = pd.read_csv(args.input_file)
    print("-" * 100)
    print("-" * 100)

    # FIRST let's have some basic idea of what this dataset contains
    print("HAVE SOME BASIC IDEA OF WHAT THIS DATASET LOOKS LIKE:")
    print()
    ds_introduction(data_file)

    # SECOND extract features from genre to make each genre not that unique,
    data_after_split_genre = extract_features(data_file, 4)
    print(data_after_split_genre)

    # THIRD scaling
    print("-" * 100)
    print("-" * 100)
    print("NOW SCALING THE DATASET TO MAKE EACH FEATURE IN SIMILAR RANGE OF VALUES")
    data_after_scaling = scaling(data_after_split_genre)
    print(data_after_scaling)

    # FOURTH feature selection by checking the pearson correlation between each feature, especially popularity
    print("-" * 100)
    print("-" * 100)
    print("NOW ENTER INTO THE FEATURE SELECTION PROCESS. LET'S CHECK THE PEARSON CORRELATION")
    print()
    print("Pearson Correlation shown in a plot")
    print()
    data_after_feature_select1 = feature_selction(data_after_scaling, 0.15, 0.5, 'option1')
    data_after_feature_select2 = feature_selction(data_after_scaling, 0.15, 0.5, 'option2')
    data_after_feature_select3 = feature_selction(data_after_scaling, 0.15, 0.5, 'option3')
    # data_after_feature_select = feature_selction(data_after_split_genre, 0.15, 0.4, 'option2')
    print("The dataset after feature selection option1 looks like: ")
    print()
    print(data_after_feature_select1)
    print()

    print("The dataset after feature selection option2 looks like: ")
    print()
    print(data_after_feature_select2)
    print()

    print("The dataset after feature selection option3 looks like: ")
    print()
    print(data_after_feature_select3)
    print()

    # FIFITH splitting into xTrain, xTest, and also yTrain, yTest
    print("-" * 100)
    print("-" * 100)
    print("SPLIT THE DATASET INTO TRAIN AND TEST DATASET BASED ON HOLDOUT METHOD")
    xTrain1, xTest1, yTrain1, yTest1 = split(data_after_scaling, data_after_feature_select1)
    print("A brief presentation of xTrain1 dataset:")
    print(xTrain1.head())
    print()
    print("A brief presentation of xTest1 dataset:")
    print(xTest1.head())
    print()

    xTrain2, xTest2, yTrain2, yTest2 = split(data_after_scaling, data_after_feature_select2)
    print("A brief presentation of xTrain2 dataset:")
    print(xTrain2.head())
    print()
    print("A brief presentation of xTest2 dataset:")
    print(xTest2.head())
    print()

    xTrain3, xTest3, yTrain3, yTest3 = split(data_after_scaling, data_after_feature_select3)
    print("A brief presentation of xTrain3 dataset:")
    print(xTrain3.head())
    print()
    print("A brief presentation of xTest3 dataset:")
    print(xTest3.head())
    print()

    # SIXTH apply the train and test dataSets into the linear_regression model
    print("-" * 100)
    print("-" * 100)
    print("APPLY THE THREE OPTIONS INTO LINEAR REGRESSION MODEL")
    linear_model1 = lr_train(xTrain1, yTrain1)
    lr_yHat1 = lr_predict(linear_model1, xTest1)
    lr_rmse1 = np.sqrt(mean_squared_error(yTest1, lr_yHat1))
    print("The root mean squared error of this option1 dataSet under linear regression model")
    print(lr_rmse1)
    print()

    linear_model2 = lr_train(xTrain2, yTrain2)
    lr_yHat2 = lr_predict(linear_model2, xTest2)
    print("The root mean squared error of this option2 dataSet under linear regression model")
    lr_rmse2 = np.sqrt(mean_squared_error(yTest2, lr_yHat2))
    print(lr_rmse2)
    print()

    linear_model3 = lr_train(xTrain3, yTrain3)
    lr_yHat3 = lr_predict(linear_model3, xTest3)
    print("The root mean squared error of this option3 dataSet under linear regression model")
    lr_rmse3 = np.sqrt(mean_squared_error(yTest3, lr_yHat3))
    print(lr_rmse3)
    print()

    plot_lr(lr_yHat1, yTest1)
    plot_lr(lr_yHat2, yTest2)
    plot_lr(lr_yHat3, yTest3)

    # SEVENTH apply the train and test dataSets into the KNN model
    print("-" * 100)
    print("-" * 100)
    print("APPLY THE THREE OPTIONS INTO DECISION TREE MODEL AND KNN MODEL")
    print("To apply decision tree model,")
    print("we need to convert the continuous value of popularity into classified labels ")
    print("Our plan is to convert all the popularity value that is above mean to be 1 and those below mean to be 0")
    print()
    mean_after_scaling = data_after_scaling['popularity'].mean()
    print("The mean of popularity value after scaling is %f" % mean_after_scaling)
    print()

    yTest1 = np.where(yTest1 >= np.mean(yTest1), 1, 0)
    yTest2 = np.where(yTest2 >= np.mean(yTest2), 1, 0)
    yTest3 = np.where(yTest3 >= np.mean(yTest3), 1, 0)

    knn_error1 = list()
    knn_error2 = list()
    knn_error3 = list()

    print("-" * 100)
    print("-" * 100)

    optimal_n = 0
    optimal_error = 0

    for i in range(1, 100):
        if i == 1:
            knn_model1 = knn_train(xTrain1, yTrain1, i)
            knn_yHat1 = knn_predict(knn_model1, xTest1)
            optimal_error = np.sqrt(mean_squared_error(yTest1, knn_yHat1))
            optimal_n = i
            knn_error1.append(optimal_error)
        else:
            knn_model1 = knn_train(xTrain1, yTrain1, i)
            knn_yHat1 = knn_predict(knn_model1, xTest1)
            knn_rmse1 = np.sqrt(mean_squared_error(yTest1, knn_yHat1))
            if optimal_error > knn_rmse1:
                optimal_error = knn_rmse1
                optimal_n = i
            knn_error1.append(knn_rmse1)

    # mse = mean_squared_error(yTest, yHat)
    print("The root mean squared error of this option1 dataSet under the optimal knn model")
    print("Optimal n-neighbors: " + str(optimal_n))
    print("RMSE: " + str(optimal_error))
    print()

    optimal_n2 = 0
    optimal_error2 = 0

    for i in range(1, 100):
        if i == 1:
            knn_model2 = knn_train(xTrain2, yTrain2, i)
            knn_yHat2 = knn_predict(knn_model2, xTest2)
            optimal_error2 = np.sqrt(mean_squared_error(yTest2, knn_yHat2))
            optimal_n2 = i
            knn_error2.append(optimal_error2)
        else:
            knn_model2 = knn_train(xTrain2, yTrain2, i)
            knn_yHat2 = knn_predict(knn_model2, xTest2)
            knn_rmse2 = np.sqrt(mean_squared_error(yTest2, knn_yHat2))
            if optimal_error2 > knn_rmse2:
                optimal_error2 = knn_rmse2
                optimal_n2 = i
            knn_error2.append(knn_rmse2)

    # mse = mean_squared_error(yTest, yHat)
    print("The root mean squared error of this option2 dataSet under the optimal knn model")
    print("Optimal n-neighbors: " + str(optimal_n2))
    print("RMSE: " + str(optimal_error2))
    print()

    optimal_n3 = 0
    optimal_error3 = 0

    for i in range(1, 100):
        if i == 1:
            knn_model3 = knn_train(xTrain3, yTrain3, i)
            knn_yHat3 = knn_predict(knn_model3, xTest3)
            optimal_error3 = np.sqrt(mean_squared_error(yTest3, knn_yHat3))
            optimal_n3 = i
            knn_error3.append(optimal_error3)
        else:
            knn_model3 = knn_train(xTrain3, yTrain3, i)
            knn_yHat3 = knn_predict(knn_model3, xTest3)
            knn_rmse3 = np.sqrt(mean_squared_error(yTest3, knn_yHat3))
            if optimal_error3 > knn_rmse3:
                optimal_error3 = knn_rmse3
                optimal_n3 = i
            knn_error3.append(knn_rmse3)

    # mse = mean_squared_error(yTest, yHat)
    print("The root mean squared error of this option3 dataSet under the optimal knn model")
    print("Optimal n-neighbors: " + str(optimal_n3))
    print("RMSE: " + str(optimal_error3))
    print()

    x_cord = range(1, 100)
    plot_knn(knn_error1, knn_error2, knn_error3)

    # EIGHTH apply the train and test dataSets into the decision tree model
    dt_error1 = list()
    dt_error2 = list()
    dt_error3 = list()

    optimal_min_leaf = 0
    optimal_max_depth = 0
    optimal_error = 0

    for i in range(2, 100):  # min-leaf-sample
        for j in range(2, 30):  # max-depth
            if i == 2 and j == 2:
                dt_model1 = dt_train(xTrain1, yTrain1, i, j)
                dt_yHat1 = dt_predict(dt_model1, xTest1)
                optimal_error = np.sqrt(mean_squared_error(yTest1, dt_yHat1))
                optimal_min_leaf = i
                optimal_max_depth = j
            else:
                dt_model1 = dt_train(xTrain1, yTrain1, i, j)
                dt_yHat1 = dt_predict(dt_model1, xTest1)
                dt_rmse1 = np.sqrt(mean_squared_error(yTest1, dt_yHat1))
                if optimal_error > dt_rmse1:
                    optimal_error = dt_rmse1
                    optimal_min_leaf = i
                    optimal_max_depth = j

    for i in range(2, 100):
        dt_model1 = dt_train(xTrain1, yTrain1, i, optimal_max_depth)
        dt_yHat1 = dt_predict(dt_model1, xTest1)
        dt_rmse1 = np.sqrt(mean_squared_error(yTest1, dt_yHat1))
        dt_error1.append(dt_rmse1)

    print("-" * 100)
    print("-" * 100)
    print("The root mean squared error of this option1 dataSet under the optimal decision tree model")
    print("Optimal min_leaf: " + str(optimal_min_leaf))
    print("Optimal max-depth: " + str(optimal_max_depth))
    print("RMSE: " + str(optimal_error))
    print()

    optimal_min_leaf2 = 0
    optimal_max_depth2 = 0
    optimal_error2 = 0

    for i in range(2, 100):  # min-leaf-sample
        for j in range(2, 30):  # max-depth
            if i == 2 and j == 2:
                dt_model2 = dt_train(xTrain2, yTrain2, i, j)
                dt_yHat2 = dt_predict(dt_model2, xTest2)
                optimal_error2 = np.sqrt(mean_squared_error(yTest2, dt_yHat2))
                optimal_min_leaf2 = i
                optimal_max_depth2 = j
            else:
                dt_model2 = dt_train(xTrain2, yTrain2, i, j)
                dt_yHat2 = dt_predict(dt_model2, xTest2)
                dt_rmse2 = np.sqrt(mean_squared_error(yTest2, dt_yHat2))
                if optimal_error2 > dt_rmse2:
                    optimal_error2 = dt_rmse2
                    optimal_min_leaf2 = i
                    optimal_max_depth2 = j

    for i_ in range(2, 100):
        dt_model2 = dt_train(xTrain2, yTrain2, i_, optimal_max_depth2)
        dt_yHat2 = dt_predict(dt_model2, xTest2)
        dt_rmse2 = np.sqrt(mean_squared_error(yTest2, dt_yHat2))
        dt_error2.append(dt_rmse2)

    print("-" * 100)
    print("-" * 100)
    print("The root mean squared error of this option2 dataSet under the optimal decision tree model")
    print("Optimal min_leaf: " + str(optimal_min_leaf2))
    print("Optimal max-depth: " + str(optimal_max_depth2))
    print("RMSE: " + str(optimal_error2))
    print()

    optimal_min_leaf3 = 0
    optimal_max_depth3 = 0
    optimal_error3 = 0

    for i in range(2, 100):  # min-leaf-sample
        for j in range(2, 30):  # max depth
            if i == 2 and j == 2:
                dt_model3 = dt_train(xTrain3, yTrain3, i, j)
                dt_yHat3 = dt_predict(dt_model3, xTest3)
                optimal_error3 = np.sqrt(mean_squared_error(yTest3, dt_yHat3))
                optimal_min_leaf3 = i
                optimal_max_depth3 = j
            else:
                dt_model3 = dt_train(xTrain3, yTrain3, i, j)
                dt_yHat3 = dt_predict(dt_model3, xTest3)
                dt_rmse3 = np.sqrt(mean_squared_error(yTest3, dt_yHat3))
                if optimal_error3 > dt_rmse3:
                    optimal_error3 = dt_rmse3
                    optimal_min_leaf3 = i
                    optimal_max_depth3 = j

    for i_ in range(2, 100):
        dt_model3 = dt_train(xTrain3, yTrain3, i_, optimal_max_depth3)
        dt_yHat3 = dt_predict(dt_model3, xTest3)
        dt_rmse3 = np.sqrt(mean_squared_error(yTest3, dt_yHat3))
        dt_error3.append(dt_rmse3)

    print("-" * 100)
    print("-" * 100)
    print("The root mean squared error of this option3 dataSet under the optimal decision tree model")
    print("Optimal min_leaf: " + str(optimal_min_leaf3))
    print("Optimal max-depth: " + str(optimal_max_depth3))
    print("RMSE: " + str(optimal_error3))
    print()

    plot_dt(dt_error1, dt_error2, dt_error3)

    # apply bagging method to have multiple linear regression models, we choose option, as
    # it's kind of in the intermediate stage
    print("-" * 100)
    print("-" * 100)
    print("FINALLY WE ENTER INTO APPLYING BAGGING METHOD")
    print("We only plan to apply ensemble method in linear regression method,")
    print("because that's the model that gives us lowest error,")
    print("and it's also the most intuitive one to fit our topic")
    print()
    print("The bagging average predicted popularity value is:")
    bagging_predict = lr_bagging(data_after_feature_select2, data_after_scaling['popularity'], 100)
    print(bagging_predict)
    print()
    print("The root mean squared of error of bagging linear regression model is: %f" % np.sqrt(mean_squared_error(data_after_scaling['popularity'], bagging_predict)))

if __name__ == "__main__":
    main()