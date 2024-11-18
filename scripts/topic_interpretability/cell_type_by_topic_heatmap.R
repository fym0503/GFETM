library(ComplexHeatmap)
library(circlize)
library(cluster)
library(svglite)
library(png)

mat <- read.table('data/kidney_margin_cell_type.csv', row.names = 1,sep=',',check.names = F,
                   header = TRUE, stringsAsFactors = FALSE)
mat = t(mat)
aster <- read.table("data/kidney_p_val_cell_type.csv",row.names = 1,sep=',',
                    header = TRUE, stringsAsFactors = FALSE)

aster = t(aster)
hmap <- Heatmap(
  mat,
  cell_fun = function(j, i, x, y, w, h, fill) {
    if (aster[i, j] < 0.012 & mat[i,j]>0.5) {
      grid.text("*", x, y,gp=gpar(fontsize=6))}},
  name='cell type',
  column_dend_side = "bottom",
  show_row_names = TRUE,
  show_column_names = TRUE,
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  row_names_gp = gpar(fontsize =5),
  column_names_gp = gpar(fontsize = 6),
  raster_quality = 5,
  rect_gp = gpar(col = "grey", lwd = 0.3),
  heatmap_legend_param = list(title_gp = gpar(fontsize = 6), labels_gp = gpar(fontsize = 6)))

mat2 <- read.table('data/kidney_margin_disease.csv', row.names = 1, sep=',', check.names = F,
                  header = TRUE, stringsAsFactors = FALSE)
mat2 = t(mat2)

aster2 <- read.table("data/kidney_p_val_disease.csv", row.names = 1, sep=',',
                     header = TRUE, stringsAsFactors = FALSE)
aster2 = t(aster2)
hmap2 <- Heatmap(
  mat2,
  cell_fun = function(j, i, x, y, w, h, fill) {
    if(aster2[i, j] < 0.01 & mat2[i,j]>0.1) {
      grid.text("*", x, y,gp=gpar(fontsize=6))}},
  name='condition',
  column_dend_side = "bottom",
  show_row_names = TRUE,
  show_column_names = TRUE,
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  row_names_gp = gpar(fontsize = 5),
  column_names_gp = gpar(fontsize =6),
  raster_quality = 5,
  rect_gp = gpar(col = "grey", lwd = 0.3),
  heatmap_legend_param = list(title_gp = gpar(fontsize = 6), labels_gp = gpar(fontsize = 6)))
pdf(file = "output/panel_c_permtest.pdf",width=3.5,height=6)
draw(hmap + hmap2, heatmap_legend_side = "right", annotation_legend_side = "right")
     
dev.off()

