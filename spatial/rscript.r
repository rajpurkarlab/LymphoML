library(spatstat)
suppressPackageStartupMessages(library("argparse"))
parser = ArgumentParser()
parser$add_argument('-p', '--patch', default='', help='enter patch id')
parser$add_argument('-n', '--num', default='', help='enter number of patches')
parser$add_argument('-t', '--tma', default='', help='enter tma id')

args = parser$parse_args()

df = read.csv(args$patch)

lb = min(as.vector(df$Location_Center_X))
print("ok")
rb = as.numeric(max(df$Location_Center_X))
ub = as.numeric(max(df$Location_Center_Y))
lowb = as.numeric(min(df$Location_Center_Y))

win = max(c(lb-rb, ub-lowb))

points = as.ppp(df[c('Location_Center_X', 'Location_Center_Y')], c(lb, rb, lowb, ub))
k = as.vector(Kest(points, r=seq(0, win/sqrt(2), 1))$iso)
id = df$group_id[1]
to.save = as.data.frame(matrix(k, nrow=1))
to.save$patch = id
write.table(to.save, paste0('/deep/group/aihc-bootcamp-fall2021/lymphoma/processed/spatial/K/tma_', args$tma, '_patch_num=', args$num, '.csv'), append=TRUE, row.names=FALSE, sep=',', col.names=FALSE)
