# Group-3-Assignment-4
## Homework Assignment 4: Monte Carlo Methods---Performance Benchmark
Griffin Arnone, Anhua Cheng, & Bailey Scoville
## Abstract
R and Python, two popular programming languages in the financial sector, rival each other in performance and accessibility. This benchmark test attempts to compare the performance of R and Python in conducting Monte Carlo simulations of the performance of four mutual funds over one year. The findings of this study can be applied to discussions of the value and effectiveness of investment fund managers. A full discussion of the benchmark test, methods, and results is included in this [paper](https://github.com/bscov/Group-3-Assignment-4/blob/main/MSDS460_Group3_HomeworkAssignment4.pdf).
## Data
We selected three actively managed mutual funds, American Funds Washington Mutual (RWMGX), Parnassus Core Equity (PRILX), T.Rowe Price U.S. Equity Research (PCCOX), and one passive ETF (WFSPX). We downloaded the historical fund closing prices since fund inception from Yahoo Finance and the CSV files are included in the [data folder](https://github.com/bscov/Group-3-Assignment-4/tree/main/Data).
## Methods
The simulation code was created using Jupyter Notebook for Python and RStudio for R. Each stage of the simulation process included lines of code to measure the quantitative metrics of execution time and memory usage. The qualitative metrics of ease of code and code efficiency were determined based on the judgment of the team. Python and R code and output are included in the [code folder](https://github.com/bscov/Group-3-Assignment-4/tree/main/Code).
## Results
The average annual returns calculated using Python and R were similar and are summarized in this [comparison table](https://github.com/bscov/Group-3-Assignment-4/blob/main/Simulated_Annual_Return_Table.csv). The ETF returns imply that actively managed mutual funds outperform the passive index fund based on the scope of our analysis. Variability plots and simulated daily return distribution plots are included in the [plots folder](https://github.com/bscov/Group-3-Assignment-4/tree/main/Plots).
Overall, R demonstrated more consistency in memory usage and Python recorded faster execution times across a variety of data inspection, statistical calculations, and simulation tasks. A full list of execution times and memory usage is included in this [table](https://github.com/bscov/Group-3-Assignment-4/blob/main/Python_R_Metrics_Table.csv).
## Conclusions
While we noticed differences in memory usage and execution time between R and Python, the differences observed in this benchmark study are minor. Based on our study, both programming languages are capable tools for performing simulation analysis.
