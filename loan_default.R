# ==============================================================================
# 0. SETUP & LIBRARIES
# ==============================================================================
# Install packages if you don't have them
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret")          # For machine learning
if(!require(arules)) install.packages("arules")        # For association rules
if(!require(cluster)) install.packages("cluster")      # For clustering
if(!require(factoextra)) install.packages("factoextra") # For cluster viz
if(!require(rpart)) install.packages("rpart")          # For Decision Trees
if(!require(rpart.plot)) install.packages("rpart.plot") # For plotting trees
if(!require(corrplot)) install.packages("corrplot")    # For correlation plots
if(!require(scales)) install.packages("scales")        # For dollar formatting

library(tidyverse)
library(caret)
library(arules)
library(cluster)
library(factoextra)
library(rpart)
library(rpart.plot)
library(corrplot)
library(scales)

# ==============================================================================
# 1. DATA LOADING & OVERVIEW
# ==============================================================================
# DATASET: 20,000 Loan Applicants
# MIXED DATA: Nominal (Gender), Ordinal (Grades), Numerical (Income, Loan Amount)
# TARGET: 'loan_paid_back' (1 = Paid, 0 = Default)

# Load Data
df <- read.csv("loan_dataset_20000.csv", stringsAsFactors = TRUE)

# Basic Inspection
cat("Dataset Dimensions:\n")
dim(df)
cat("\nOverview of Data:\n")
glimpse(df)

# ==============================================================================
# 2. PREPROCESSING & EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================

# --- A. Data Cleaning ---
# Check for missing values
sum(is.na(df))

# Convert Target 'loan_paid_back' to a Factor for Classification
# 1 = Paid, 0 = Default
df$loan_paid_back <- factor(df$loan_paid_back, levels = c(0, 1), labels = c("Default", "Paid"))

# --- B. Visualization (Enhanced) ---

# Plot 1: Target Distribution (Imbalance Check)
# WHY: To see if we have enough examples of 'Default' to train a model.
ggplot(df, aes(x = loan_paid_back, fill = loan_paid_back)) +
  geom_bar() +
  labs(title = "Distribution of Loan Status", x = "Status", y = "Count") +
  theme_minimal()

# Plot 2: Credit Score vs. Loan Status
# WHY: To test the hypothesis that higher credit scores lead to fewer defaults.
ggplot(df, aes(x = loan_paid_back, y = credit_score, fill = loan_paid_back)) +
  geom_boxplot() +
  labs(title = "Credit Score Distribution by Loan Outcome", y = "Credit Score") +
  theme_minimal()

# Graph 1: Who pays back the most? (Loan Repayment Rate by Employment)
# We calculate the % of people who paid back in each group
df_emp_rate <- df %>%
  group_by(employment_status) %>%
  summarise(repayment_rate = mean(loan_paid_back == "Paid"))

ggplot(df_emp_rate, aes(x = reorder(employment_status, repayment_rate), y = repayment_rate)) +
  geom_col(fill = "#3498DB", width = 0.6) +
  geom_text(aes(label = scales::percent(repayment_rate, accuracy = 0.1)), vjust = -0.5) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(title = "Loan Repayment Rate by Employment Status",
       subtitle = "employed people are slightly more reliable than self-employed",
       x = "Employment Status", y = "Repayment Rate") +
  theme_minimal()

# Graph 2: Do educated people have better Credit Scores?
# Average Credit Score by Education Level
df_edu_score <- df %>%
  group_by(education_level) %>%
  summarise(avg_score = mean(credit_score))

ggplot(df_edu_score, aes(x = reorder(education_level, avg_score), y = avg_score)) +
  geom_col(fill = "#2ECC71", width = 0.6) +
  coord_cartesian(ylim = c(600, 750)) + # Zoom in to see differences
  geom_text(aes(label = round(avg_score, 0)), vjust = -0.5) +
  labs(title = "Average Credit Score by Education Level",
       x = "Education Level", y = "Average Credit Score") +
  theme_minimal()

# Graph 3: Demographics Credit Analysis (Gender & Marital Status)
# We use a boxplot to show the spread of credit scores across groups
ggplot(df, aes(x = marital_status, y = credit_score, fill = gender)) +
  geom_boxplot() +
  labs(title = "Credit Score Distribution: Gender & Marital Status",
       subtitle = "Check if specific demographic groups have lower scores",
       x = "Marital Status", y = "Credit Score") +
  theme_minimal() +
  theme(legend.position = "top")


# --- C. The Risk Matrix (Heatmap) ---

# Graph 4: Employment x Marital Status Risk Matrix
# We calculate the DEFAULT RATE (Risk) for every combination
risk_matrix <- df %>%
  group_by(employment_status, marital_status) %>%
  summarise(default_rate = mean(loan_paid_back == "Default"))

ggplot(risk_matrix, aes(x = marital_status, y = employment_status, fill = default_rate)) +
  geom_tile(color = "white") +
  geom_text(aes(label = scales::percent(default_rate, accuracy = 0.1)), color = "black") +
  scale_fill_gradient(low = "#D6EAF8", high = "#E74C3C") + # Blue (Safe) to Red (Risky)
  labs(title = "Risk Matrix: Probability of Default",
       subtitle = "Redder cells indicate higher risk groups",
       x = "Marital Status", y = "Employment Status", fill = "Default Risk") +
  theme_minimal()

# --- D. Correlation Matrix ---
# 1. Select numeric variables but REMOVE the ones you don't want
numeric_vars <- df %>% 
  select_if(is.numeric) %>%
  select(-age, -public_records, -num_of_open_accounts, -monthly_income)

# 2. Calculate Correlation
corr_matrix <- cor(numeric_vars)

# 3. Create the Plot (No Numbers, Just Colors)
col_palette <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))

corrplot(corr_matrix, 
         method = "color", 
         col = col_palette(200),  
         type = "upper", 
         order = "hclust", 
         
         # REMOVED: addCoef.col = "black" (This removes the numbers)
         
         tl.col = "black", tl.srt = 45, # Keep text labels black and rotated
         diag = FALSE,                  # Hide the diagonal line
         title = "Correlation Heatmap",
         mar = c(0,0,1,0))