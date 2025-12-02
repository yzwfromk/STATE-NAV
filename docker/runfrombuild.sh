#!/bin/bash
IMAGE_NAME="twoleggedcoffeedrinker/state_nav:latest"
HOST_HOME_DIR=$HOME
catkin_ws="$HOST_HOME_DIR/Desktop/Ziwon_Project/ros1_ws" # Change this to your catkin workspace directory

# Define environment variables for enabling graphical output for the container.
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth

if [ -e $XAUTH ]  # Check if it exists (either file or directory)
then
    if [ -d $XAUTH ]  # Check if it's a directory
    then
        rm -r $XAUTH  # Remove the directory
    else
        rm $XAUTH  # Remove the file
    fi
fi
touch $XAUTH
xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
chmod a+r $XAUTH



#==
# Launch container
#==

# Create symlinks to user configs within the build context.
mkdir -p .etc && cd .etc
ln -sf /etc/passwd .
ln -sf /etc/shadow .
ln -sf /etc/group .
cd ..

# Launch a container from the prebuilt image.
echo "---------------------"
RUN_COMMAND="docker run \
  --volume=$XSOCK:$XSOCK:rw \
  --volume=$XAUTH:$XAUTH:rw \
  --env="QT_X11_NO_MITSHM=1" \
  --env="XAUTHORITY=$XAUTH" \
  --env="DISPLAY=$DISPLAY" \
  --ulimit rtprio=99 \
  --cap-add=sys_nice \
  --privileged \
  --net=host \
  --entrypoint /bin/bash \
  -eHOST_USERNAME=$(whoami) \
  --env HOST_HOME_DIR=$HOST_HOME_DIR \
  --env catkin_ws=$catkin_ws \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env NVIDIA_VISIBLE_DEVICES=all \
  --env LD_LIBRARY_PATH=/usr/local/zed/lib:\$LD_LIBRARY_PATH \
  -v$HOST_HOME_DIR:$HOST_HOME_DIR \
  -v$(pwd)/.etc/shadow:/etc/shadow \
  -v$(pwd)/.etc/passwd:/etc/passwd \
  -v$(pwd)/.etc/group:/etc/group \
  -v/media:/media \
  -v/dev:/dev \
  --gpus all \
  --cgroupns=host \
  -v /sys/fs/cgroup:/sys/fs/cgroup:rw \
  -e ROS_DOMAIN_ID=0 \
  -it $IMAGE_NAME"
echo -e "[run.sh]: \e[1;32mThe final run command is\n\e[0;35m$RUN_COMMAND\e[0m."
$RUN_COMMAND
echo -e "[run.sh]: \e[1;32mDocker terminal closed.\e[0m"
#   --entrypoint=$ENTRYPOINT \