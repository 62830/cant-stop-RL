import game
from tqdm import tqdm
from fraction_eval import Fraction_Eval_Agent, Fraction_Eval_Agent_v2
from game import Agent

# 95% confidence one-sided z-test
def hypothesis_test(n, p, z = 1.645):
    h_std = (0.5 * 0.5 / n) ** 0.5
    z_score = (p - 0.5) / h_std
    return z_score > z, z_score


class Duel:
    def __init__(self, agents):
        self.agents = agents

    def play(self, num_games = 1000, record = False, verbose = False, swap_first = False, progress = False):
        if swap_first:
            w01 = self._play(num_games // 2, record, verbose, progress)
            self.agents = self.agents[::-1]
            w10 = self._play(num_games // 2, record, verbose, progress)
            self.agents = self.agents[::-1]
            return w01, w10[::-1]
        else:
            return self._play(num_games, record, verbose)

    def _play(self, num_games = 500, record = False, verbose = False, progress = False):
        win_count = [0] * len(self.agents)
        iter = range(num_games)
        g = game.Game(self.agents)
        if progress:
            iter = tqdm(iter)
        for _ in iter:
            score = g.play(record, verbose)
            win_count[score.index(max(score))] += 1
            g.reset()

        if verbose:
            print('Win count:', win_count)
        return win_count
    
if __name__ == '__main__':
    agents = [Fraction_Eval_Agent_v2(), Fraction_Eval_Agent_v2()]
    duel = Duel(agents)
    show = True
    if show:
        num_games = 1
        win_count = duel.play(num_games = num_games, record = False, verbose = True, swap_first = False, progress = False)
    else:
        num_games = 2000
        win_count = duel.play(num_games = num_games, record = False, verbose = False, swap_first = True, progress = True)
        # hypothesis test
        n = num_games
        p = (win_count[0][0]+win_count[1][0]) / n
        print('Agent 0 win rate:', p)
        passed, z = hypothesis_test(n, p)
        if passed:
            print('Agent 0 wins with confidence')
        else:
            print('Agent 0 does not win with confidence')
