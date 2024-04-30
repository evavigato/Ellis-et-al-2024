library (gplots)
data <- read.table("Connectivity_data_for_R.csv", header = TRUE, sep = ",")
heatmap_data <- as.matrix(data[, -1])

dist_matrix <- dist(heatmap_data)
hclust_result <- hclust(dist_matrix)
dend <- as.dendrogram(hclust_result)

colors_rev <- colorRampPalette(rev(heat.colors(256)))

heatmap.2(heatmap_data, Rowv = dend, col = colors_rev, scale = "none",
          main = "Heatmap with Dendrogram", xlab = "Species", ylab = "Glomeruli",
          trace = "none", key = TRUE, symm = FALSE,
          labRow = data$Glomerulus, cexRow = 0.5, labCol = colnames(heatmap_data), cexCol = 1)




