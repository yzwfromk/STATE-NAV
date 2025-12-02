echo "Running initial setup..."

# echo 'catkin_ws="$HOST_HOME_DIR/Desktop/Ziwon_Project/ros1_ws"' >> ~/.bashrc
echo 'source /opt/ros/noetic/setup.bash' >> ~/.bashrc
echo 'source $catkin_ws/devel/local_setup.bash' >> ~/.bashrc
echo 'source $catkin_ws/devel/setup.bash' >> ~/.bashrc
source ~/.bashrc


echo "Host's home directory: $HOST_HOME_DIR"
echo "catkin_ws: $catkin_ws"
source ~/.bashrc



# STATE-NAV
echo -e "Installing STATE-NAV..."
cd $catkin_ws/src/state_nav 
pip install -r requirements.txt
pip install -e .



# ROS build
echo -e "Building ROS workspace..."
cd $catkin_ws
catkin build
source ~/.bashrc


