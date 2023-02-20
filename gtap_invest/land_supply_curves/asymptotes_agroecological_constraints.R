####################################################################################################################################

# Identifying areas biophyscially unsuitable for agriculture 

library(raster)
library(sf)

m <- c(0, 6, NA,  
       6, 8, 1,
       8, 9, NA)

(rclmat <- matrix(m, ncol=3, byrow=TRUE))

setwd("E:/GAEZ_IIASA/all_other_crops/Cereals/Wheat")

Wheat <- raster("data.asc")
Wheat <- reclassify(Wheat, rclmat)

setwd("E:/GAEZ_IIASA/all_other_crops/Cereals/Dryland_Rice")

Dryland_Rice <- raster("data.asc")
Dryland_Rice <- reclassify(Dryland_Rice, rclmat)

setwd("E:/GAEZ_IIASA/all_other_crops/Cereals/Wetland_Rice")

Wetland_Rice <- raster("data.asc")
Wetland_Rice <- reclassify(Wetland_Rice, rclmat)

setwd("E:/GAEZ_IIASA/all_other_crops/Cereals/Barley")
Barley <- raster("data.asc")
Barley <- reclassify(Barley, rclmat)

setwd("E:/GAEZ_IIASA/all_other_crops/Cereals/Maize")
Maize <- raster("data.asc")
Maize <- reclassify(Maize, rclmat)

setwd("E:/GAEZ_IIASA/all_other_crops/Roots_Tubers/White_Potato")
White_Potato <- raster("data.asc")
White_Potato <- reclassify(White_Potato, rclmat)

setwd("E:/GAEZ_IIASA/all_other_crops/Roots_Tubers/Sweet_Potato")
Sweet_Potato <- raster("data.asc")
Sweet_Potato <- reclassify(Sweet_Potato, rclmat)

setwd("E:/GAEZ_IIASA/all_other_crops/Sugar_Crops/Sugarcane")
Sugarcane <- raster("data.asc")
Sugarcane <- reclassify(Sugarcane, rclmat)

setwd("E:/GAEZ_IIASA/all_other_crops/Fruits/Banana")
Banana <- raster("data.asc")
Banana <- reclassify(Banana, rclmat)

setwd("E:/GAEZ_IIASA/all_other_crops/Vegetables/Tomato")
Tomato <- raster("data.asc")
Tomato <- reclassify(Tomato, rclmat)

Crops <- stack(Banana, Barley, Dryland_Rice, Maize, Sugarcane, Sweet_Potato, Tomato, Wetland_Rice, Wheat, White_Potato)
Crops <- calc(Crops, sum) #takes < 30 seconds

rm(list = setdiff(ls(), c("Crops", "rclmat")))

setwd("E:/GAEZ_IIASA/all_oil_crops/coconut_high")
coconut <- raster("data.asc")
coconut <- reclassify(coconut, rclmat)

setwd("E:/GAEZ_IIASA/all_oil_crops/cotton_high")
cotton <- raster("data.asc")
cotton <- reclassify(cotton, rclmat)

setwd("E:/GAEZ_IIASA/all_oil_crops/groundnut_high")
groundnut <- raster("data.asc")
groundnut <- reclassify(groundnut, rclmat)

setwd("E:/GAEZ_IIASA/all_oil_crops/jatropha_high")
jatropha <- raster("data.asc")
jatropha <- reclassify(jatropha, rclmat)

setwd("E:/GAEZ_IIASA/all_oil_crops/oil_palm_high")
oil_palm <- raster("data.asc")
oil_palm <- reclassify(oil_palm, rclmat)

setwd("E:/GAEZ_IIASA/all_oil_crops/rapeseed_high")
rapeseed <- raster("data.asc")
rapeseed <- reclassify(rapeseed, rclmat)

setwd("E:/GAEZ_IIASA/all_oil_crops/soybean_high")
soybean <- raster("data.asc")
soybean <- reclassify(soybean, rclmat)

setwd("E:/GAEZ_IIASA/all_oil_crops/sunflower_high")
sunflower <- raster("data.asc")
sunflower <- reclassify(sunflower, rclmat)

oil_crops <- stack(coconut, cotton, groundnut, jatropha, oil_palm,
                   rapeseed, soybean, sunflower)

oil_crops <- calc(oil_crops, sum) 

rm(list = setdiff(ls(), c("Crops", "oil_crops")))

Crops[Crops==10] <- 1
oil_crops[oil_crops==8] <- 1

Crops[is.na(Crops)] <- 0
oil_crops[is.na(oil_crops)] <- 0

Unsuitable_crops <- stack(Crops, oil_crops)
Unsuitable_crops <- calc(Unsuitable_crops, sum) 

m <- c(0, 1, NA,  
       1, 2, 1)
(rclmat <- matrix(m, ncol=3, byrow=TRUE))
Unsuitable_crops <- reclassify(Unsuitable_crops, rclmat)
Unsuitable_crops[Unsuitable_crops==0] <- NA

setwd("E:/Land_supply_asymptotes/GAEZ_crop_analysis")
writeRaster(Unsuitable_crops, "Unsuitable_crops.tif", overwrite = T)
#unlink("Unsuitable_crops.tif")
