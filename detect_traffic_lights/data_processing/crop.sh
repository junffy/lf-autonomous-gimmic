source=crop_traffic_light.py
echo traffic lights crop processing...
for f in *.png *.PNG *.jpg *.jpeg *.JPG *.JPEG 
do
    if [ -f $f ]; then
        echo python $source $f
        python $source $f
    fi
done
echo done
