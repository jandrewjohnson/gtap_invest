import os
import hazelbean as hb

import gtap_invest_generate_base_data as gti_bd

p = hb.ProjectFlow('../../projects/generate_base_data')
p.gtap37_aez18_input_vector_path = os.path.join(p.model_base_data_dir, "region_boundaries\GTAPv10_AEZ18_37Reg.shp")

joined_region_vectors_task = p.add_task(gti_bd.joined_region_vectors)
joined_region_vectors_task.run = 0

gtap_vector_pyramid_task = p.add_task(gti_bd.gtap_vector_pyramid)
gtap_vector_pyramid_task.run = 1

p.execute()

