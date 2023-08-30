
import random

def pi_estimation(n=100_000):
    circle_points = 0
    square_points = 0
    for i in range(n):
        rand_x = random.uniform(-1, 1)
        rand_y = random.uniform(-1, 1)
        origin_dist = rand_x ** 2 + rand_y ** 2
        if origin_dist <= 1:
            circle_points += 1
        square_points += 1
    return 4 * circle_points / square_points

