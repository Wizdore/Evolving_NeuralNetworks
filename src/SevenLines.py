import gym, numpy
ENV = gym.make('CartPole-v0')
# WHO NEEDS CONDITIONAL STATEMENTS!!?? Or programs bigger than 8 lines??
state = numpy.array([0.69133231, -0.20516511, -0.75820921, 0.26613669, 0.27673464, -0.36717599,-0.26046508, -0.28477222, 0.52321166, -0.80721378, 0.29428328, 0.29825771,0.80397947, -0.326007, -0.34943436, -0.57278871, -0.98233225, -0.84820036,-0.52714673, -0.57732037, 0.99911137, -0.84907632, 0.49275814, -0.89989096])
observation = ENV.reset()
while True:
    ENV.render()
    observation = ENV.step(int(numpy.round(((numpy.matmul((numpy.matmul((2 / (1 + numpy.exp(-0.5 * observation))) - 1, state[0:16].reshape(4, 4)) + state[16:20]), state[20:24].reshape(4, 1)))+1)/2)))[0]
