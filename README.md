# wetnet
Final project for Brown's CS1470 course, using style transfer methods to upsample 2D water simulations.

To set up the virtual environment, run setup-virtual-environment.sh FROM THE VIRTUAL-ENV DIRECTORY.

To activate the virtual environment, run source virtual-env/wetnet-env/bin/activate.


# To test on Smoke style transfer paper, run:

python wetnet/app/app.py to generate density data.
cd smoke_style_transfer
python3 styler2d.py --tag net --content_layer mixed3b_3x3_bottleneck_pre_relu --content_channel 44
