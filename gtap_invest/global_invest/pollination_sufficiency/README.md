# pollination_sufficiency
Going back to basics, this is just the landscape pollination sufficiency parts. 

make_poll_suff.py maps the pollination sufficiency based on configuration of crop and noncrop habitat in a LULC map
(this used to be called dasgupta_agriculture.py but then we did it for more than dasgupta)

realized_pollination.py maps pollinated production back to habitat, using the reverse of pollination sufficiency

the pollination dependence part was done in another code base but until that Monfreda crop data gets updated, there's no point to re-running it each time. Those rasters are here:
https://storage.googleapis.com/ecoshard-root/ipbes/monfreda_pollination_dependence.zip
