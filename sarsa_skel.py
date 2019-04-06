""" Exemplu de cod pentru tema 1.
"""

import matplotlib.pyplot as plt
import numpy as np
import gym
import math
from tqdm import tqdm
import gym_minigrid  # pylint: disable=unused-import
from argparse import ArgumentParser

def softmax(Q, state, N, c, env, args):

	beta = 0

	if (N.get(state, None) is None):
		N[state] = 0

	actions = [(a1, a2) for (i1, a1) in enumerate(env.actions) for (i2, a2) in enumerate(env.actions) if a1 != a2 and i2 > i1]

	max = None

	if N[state] > 0:
		for (a1, a2) in actions:
			differece = abs(Q.get((state, a1), 0) - Q.get((state, a2), 0))
			if max is None or max < differece:
				max = differece
		
		if(max == 0):
			beta = 1
		else:
			beta = math.log(N[state]) / float(max)

	# sum_qa = sum([math.exp(beta * Q.get((state, action), 0)) for action in env.actions])
	sum_qa = 0.0

	for action in env.actions:
		sum_qa += math.exp(beta * Q.get((state, action), 0))

	probabilities = [math.exp(beta * Q.get((state, action), 0)) / float(sum_qa) for action in env.actions]

	return np.random.choice(list(env.actions), p = probabilities)

def epsilon_greedy(Q, state, N, c, env, args):

	maxActions = []
	
	if(N.get(state, None) is None):
		N[state] = 0
	
	if(N[state] != 0):
		epsilon = float(c) / N[state]
	else:
		epsilon = 1

	actionToValue = {}

	for action in env.actions:
		actionToValue[action] = Q.get((state, action), 0)

	maxValue = max(actionToValue.values())
	actionCount = len(actionToValue.keys())

	deltaDifference = 0.0001

	maxActions = [action for action, value in actionToValue.items() if abs(maxValue - value) < deltaDifference]

	probabilities = [(float(epsilon) / actionCount + float(1 - epsilon) / len(maxActions)) if action in maxActions else float(epsilon) / actionCount for action, value in actionToValue.items()]
	
	# print(actionToValue.keys())
	return np.random.choice(list(actionToValue.keys()), p = probabilities)

def main():
	""" Exemplu de evaluare a unei politici pe parcursul antrenării (online).
	"""

	parser = ArgumentParser()

	# Learning rate
	parser.add_argument("--learning_rate", type=float, default=0.1,
						help="Learning rate -> Alpha")
	
	#Discount
	parser.add_argument("--discount", type=float, default=0.9,
						help="Discount -> Gamma.")

	#Epochs
	parser.add_argument("--epochs", type=int, default=5000,
		help="Number of epochs")

	#Report frequency
	parser.add_argument("--report_frequency", type=int, default=2000,
		help="Result report frequency")

	parser.add_argument("--c", type=float, default=0.05,
                        help="Epsilon greedy constant.")
	#Environment type
	parser.add_argument("--environment", type=str, default="MiniGrid-Empty-6x6-v0",
		help="Environment type")

	#Strategy
	parser.add_argument("--strategy", type=str, default="e-greedy",
		help="Strategy. Possible values : softmax or e-greedy")

	#Default Q value
	parser.add_argument("--q0", type=int, default=0,
		help="Default Q value")

	#Plot image file name
	parser.add_argument("--file_name", type=str, default="file",
		help="Plot image file name")

	

	args = parser.parse_args()

	env = gym.make(args.environment)

	# renderer = env.render("human")

	#declare Q and N
	Q = {}
	N = {}

	steps, avg_returns, avg_lengths = [], [], []
	recent_returns, recent_lengths = [], []
	crt_return, crt_length = 0, 0

	state, done = env.reset(), False

	#initialize frequency for the first state

	# str_state = str(state)
	str_state = str(env)

	for step in range(1, args.epochs + 1):

		# print(step)

		if args.strategy == "softmax":
			action = softmax(Q, str_state, N, args.c, env, args)
		else:
			action = epsilon_greedy(Q, str_state, N, args.c, env, args)

		while (done == False):

			if(N.get(str_state, None) is None):
				N[str_state] = 0
			else:
				N[str_state] += 1

			next_state, reward, done, _ = env.step(action)

			str_next_state = str(env)

			crt_return += reward
			crt_length += 1

			if args.strategy == "softmax":
				next_action = softmax(Q, str_next_state, N, args.c, env, args)
			else:
				next_action = epsilon_greedy(Q, str_next_state, N, args.c, env, args)

			if(Q.get((str_state, action), None) is None):
				Q[(str_state, action)] = args.q0
			
			if(Q.get((str_next_state, next_action), None) is None):
				Q[(str_next_state, next_action)] = args.q0

			Q[(str_state, action)] +=  args.learning_rate * (reward + args.discount * Q[(str_next_state, next_action)] - Q[(str_state, action)])

			state = next_state
			action = next_action
			str_state = str_next_state
			# env.render("human")

		#After the epoch reset the state
		# if crt_return > 0:
		# 	print(str(step) + " -> victory!")
		# else:
		# 	print(str(step) + " -> lost")

		state, done = env.reset(), False
		recent_returns.append(crt_return)  # câștigul episodului încheiat
		recent_lengths.append(crt_length)
		crt_return, crt_length = 0, 0
		
		if step % args.report_frequency == 0:
			avg_return = np.mean(recent_returns)  # media câștigurilor recente
			avg_length = np.mean(recent_lengths)  # media lungimilor episoadelor

			steps.append(step)  # pasul la care am reținut valorile
			avg_returns.append(avg_return)
			avg_lengths.append(avg_length)

			print(  # pylint: disable=bad-continuation
				f"Step {step:4d}"
				f" | Avg. return = {avg_return:.2f}"
				f" | Avg. ep. length: {avg_length:.2f}"
			)

			recent_returns.clear()
			recent_lengths.clear()

	# La finalul antrenării afișăm evoluția câștigului mediu
	# În temă vreau să faceți media mai multor astfel de traiectorii pentru
	# a nu trage concluzii fără a lua în calcul varianța algoritmilor

	file_name = str(args.environment) + "_" + str(args.strategy) + "_" + str(args.learning_rate) + "_" + str(args.c) + "_" + str(args.epochs)

	description = "Map=" + args.environment + " Strategy=" + args.strategy + " Alpha=" + str(args.learning_rate) + " Gamma=" + str(args.discount) + " C=" + str(args.c) + " Epochs=" + str(args.epochs)
	
	
	# plt.figure(figsize=(15, 5))
	# plt.suptitle("Muie steaua")
	# plt.subplot(131)
	# plt.title("Average episode length")
	# plt.plot(steps, avg_returns, c='black', label='SAMME')

	# plt.legend()
	# plt.ylim(0.18, 0.62)
	# plt.ylabel('Test Error')
	# plt.xlabel('Number of Trees')

	# plt.subplot(132)
	# plt.title("Average episode return")
	# plt.plot(steps, avg_lengths, c='black', label='SAMME')

	_fig, (ax1, ax2) = plt.subplots(ncols=2)
	ax1.plot(steps, avg_lengths, label=args.strategy)
	plt.suptitle(description, fontsize = 7)
	
	ax1.set_title("Average episode length")
	ax1.legend()

	ax2.plot(steps, avg_returns, label=args.strategy)
	ax2.set_title("Average episode return")
	ax2.legend()
	plt.savefig(file_name, format='pdf')

if __name__ == "__main__":
	main()
