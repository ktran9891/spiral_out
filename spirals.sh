#!/bin/sh


#for seed_image in './experimental_results/ktran.jpg' './experimental_results/maynard_james_keenan.jpeg' './experimental_results/gaius_baltar.png' './experimental_results/jon_snow.jpeg'; do
for seed_image in './experimental_results/maynard_james_keenan.jpeg' './experimental_results/gaius_baltar.png' './experimental_results/jon_snow.jpeg'; do
    for attribute in 'beautiful' 'broken' 'elegant' 'pensive'; do
        python spiral_out.py $seed_image $attribute
    done
done
