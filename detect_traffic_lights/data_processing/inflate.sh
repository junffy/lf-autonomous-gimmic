source=crop.py

echo processing...
for f in *.jpg *.jpeg *.JPG *.JPEG *.png *.PNG
do
	if [ -f $f ]; then
		echo python $source $f
		python $source $f
	fi
done

echo done