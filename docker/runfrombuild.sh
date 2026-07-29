#!/bin/bash
IMAGE_NAME="biped_nav_sim_updated:latest"
HOST_HOME_DIR=$HOME
AUTO_SETUP="${AUTO_SETUP:-1}"
# HOST_HOME_DIR=/home/lidar # Change this to your home directory

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
if [ "$AUTO_SETUP" = "1" ]; then
  CONTAINER_SETUP_CMD=". \"${HOST_HOME_DIR}/ros2_ws/statenav_ws/src/state_nav/docker/startup.sh\" && source \"${HOST_HOME_DIR}/ros2_ws/statenav_ws/install/setup.bash\" && exec bash -i"
else
  CONTAINER_SETUP_CMD="exec bash -i"
fi
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
  --env AUTO_SETUP=$AUTO_SETUP \
  --env __GLX_VENDOR_LIBRARY_NAME=nvidia \
  --env __NV_PRIME_RENDER_OFFLOAD=1 \
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
  -it $IMAGE_NAME \
  -lc '$CONTAINER_SETUP_CMD'"
echo -e "[run.sh]: \e[1;32mThe final run command is\n\e[0;35m$RUN_COMMAND\e[0m."
eval "$RUN_COMMAND"
echo -e "[run.sh]: \e[1;32mDocker terminal closed.\e[0m"
#   --entrypoint=$ENTRYPOINT \
