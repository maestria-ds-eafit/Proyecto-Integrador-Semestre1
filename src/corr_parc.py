import pingouin as pg

def partial_correlation(df,x,y,others):
    return pg.partial_corr(data=df, x=x, y=y, covar=others, method='pearson')
