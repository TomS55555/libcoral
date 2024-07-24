# C++ examples using Edge TPU

To build all the examples in this directory, you first need to
[install Bazel](https://docs.bazel.build/versions/master/install.html) and
(optional but we recommend)
[Docker](https://docs.docker.com/install/).

Then navigate up to the root `libcoral` directory and run the following command:

```
make DOCKER_IMAGE=debian:stretch DOCKER_CPUS="aarch64" DOCKER_TARGETS="examples" docker-build
```

When done, you'll find the example binaries in `libcoral/out/aarch64/examples/`.

The above command builds for `aarch64` (compatible with the Coral Dev Board),
but alternative CPU options are `k8`, `armv7a`, and `darwin`.

**Tip:** Instead of building on your computer, just
[run this Colab notebook](https://colab.sandbox.google.com/github/google-coral/tutorials/blob/master/build_cpp_examples.ipynb)
to build the examples and download the binaries.

# SUPERSORT TPU build instructions (docker)

The easiest way to compile the code is with the following comamnd:
`make DOCKER_IMAGE=debian:buster DOCKER_CPUS="aarch64" DOCKER_TARGETS=examples docker-build`

However, for developping it is easier to build the docker container and compile inside the container instead of rebuilding it every time:

`docker run -i --tty -v /home/tom/libcoral/:/workspace --name 'my-coral-container' "coral-edgetpu-debian-buster"`:  
* -i for interactive 
* --tty for terminal
* -v to mount the host directory to the container
* --name to give a name to the container
* "coral-edgetpu-debian-buster" is the name of the image, make sure it is in the docker images list  

If the container is already built, you can search for it by id or by name with  
`docker ps -a`  
You can then start the container with the following command:  
    `docker start my-coral-container`  
Then you can enter the container with the following command:  
    `docker exec -it my-coral-container /bin/bash`  
Finally you can compile the code with the following command:  
    `make CPU="aarch64" COMPILATION_MODE=opt -C workspace/ examples`