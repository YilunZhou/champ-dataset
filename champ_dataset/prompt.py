
import random
from copy import deepcopy as copy
from functools import partial
from . import *

sys_prompt = 'You are an expert on mathematics.'

prompt_main = '\n\n'.join([
    'Solve the following problem. Make sure to show your work before giving the final answer.', 
    '{problem.text}'
])

prompt_ch_list = '\n\n'.join([
    prompt_main, 
    'You may find the following information useful:',
    '{ch_list_str}'
])

prompt_name = 'Please explain the following concept: {concept.name}.'

prompt_example = '\n\n'.join([
    'Please give an example that applies the following concept:', 
    '{concept.text}'
])

prompt_name_apply_has_remaining_ch = '\n\n'.join([
    prompt_main, 
    'Besides the concept{plural_s} above, you may also find the following information useful:', 
    '{rem_ch_list_str}'
])

prompt_name_apply_no_remaining_ch = '\n\n'.join([
    prompt_main, 
    'You may find the above concept{plural_s} helpful.'
])

prompt_problem_single = '\n\n'.join([
    'First, study the following problem and its solution as they may contain helpful information.', 
    '{sample_problem}'
])

prompt_problem_multiple = '\n\n'.join([
    'First, study the following problems and their solutions as they may contain helpful information.',
    '{sample_problems}'
])

prompt_problem_query = ('\n\nWith what you have learned, solve the following problem. '
                 'Make sure to show your work before giving the final answer.\n\n{problem.text}')

prompt_problem_remaining = ('\n\nIn addition, you may also find the following information helpful:'
                '\n\n{rem_ch_list_str}')


prompt_partial_soln = '\n\n'.join([
    prompt_main, 
    'Below is a partial solution to the problem that may be helpful:', 
    '{soln_list_str}'
])

prompt_summarizer = ('Now, summarize the answer above in one sentence, '
                     'without any intermediate steps or explanations.')

def plural_s(i):
    assert i != 0 and isinstance(i, int)
    return '' if i == 1 else 's'

class PromptGenerator():
    def __init__(self, dataset):
        self.dataset = dataset

    def show_solution(self, pid):
        p = self.dataset[pid]
        result = f'Problem: {p.text}\nStep-wise solution:\n'
        result += self.enumerate_solution(p.solution.steps)
        return result

    def enumerate_solution(self, steps):
        return '\n'.join([f'{idx+1}. {step.text}' for idx, step in enumerate(steps)])

    def enumerate_ch(self, ch_list):
        assert len(ch_list) != 0
        return '\n'.join([f'{idx+1}. {self.dataset[ch].text}' for idx, ch in enumerate(ch_list)])

    def problem_only(self, problem):
        return sys_prompt, [prompt_main.format(problem=problem)], []

    def problem_with_ch_list(self, problem, ch_list):
        if len(ch_list) == 0:
            return self.problem_only(problem)
        ch_list_str = self.enumerate_ch(ch_list)
        return sys_prompt, [prompt_ch_list.format(problem=problem, ch_list_str=ch_list_str)], []

    def no_c(self, problem, with_h):
        if with_h:
            ch_list = [ch for ch in problem.ch_list if ch.startswith('H')]
        else:
            ch_list = []
        return self.problem_with_ch_list(problem, ch_list)

    def direct_c(self, problem, with_h):
        if with_h:
            ch_list = problem.ch_list
        else:
            ch_list = [ch for ch in problem.ch_list if ch.startswith('C')]
        return self.problem_with_ch_list(problem, ch_list)

    def name_c(self, problem, with_h):
        prompts = []
        remaining_chs = []
        for ch in problem.ch_list:
            if ch.startswith('C'):
                if self.dataset[ch].name is not None:
                    prompts.append(prompt_name.format(concept=self.dataset[ch]))
                else:
                    remaining_chs.append(ch)
            else:
                if with_h:
                    remaining_chs.append(ch)
        if len(prompts) == 0:
            return self.direct_c(problem, with_h)
        if len(remaining_chs) != 0:
            query_prompt = prompt_name_apply_has_remaining_ch.format(
                problem=problem, 
                plural_s=plural_s(len(prompts)), 
                rem_ch_list_str=self.enumerate_ch(remaining_chs))
        else:
            query_prompt = prompt_name_apply_no_remaining_ch.format(
                problem=problem, 
                plural_s=plural_s(len(prompts)))
        prompts.append(query_prompt)
        return sys_prompt, prompts, []

    def example_c(self, problem, with_h):
        prompts = []
        for ch in problem.ch_list:
            if ch.startswith('C'):
                prompts.append(prompt_example.format(concept=self.dataset[ch]))
        if len(prompts) == 0:
            return self.direct_c(problem, with_h)
        if with_h:
            hs = [ch for ch in problem.ch_list if ch.startswith('H')]
        else:
            hs = []
        if len(hs) != 0:
            query_prompt = prompt_name_apply_has_remaining_ch.format(
                problem=problem, 
                plural_s=plural_s(len(prompts)), 
                rem_ch_list_str=self.enumerate_ch(hs))
        else:
            query_prompt = prompt_name_apply_no_remaining_ch.format(
                problem=problem, 
                plural_s=plural_s(len(prompts)))
        prompts.append(query_prompt)
        return sys_prompt, prompts, []

    def root_c(self, problem, with_h):
        ch_instances = [self.dataset[ch] for ch in problem.ch_list]
        if with_h:
            ch_list = [ch.get_root().identifier if isinstance(ch, Concept) else ch.identifier 
                for ch in ch_instances]
        else:
            ch_list = [ch.get_root().identifier for ch in ch_instances if isinstance(ch, Concept)]
        return self.problem_with_ch_list(problem, ch_list)

    def build_c2p_dict(self):
        try:
            self.c2p
        except AttributeError:
            self.c2p = dict()
            for pid, p in self.dataset.problems.items():
                for ch in p.ch_list:
                    if ch.startswith('H'):
                        continue
                    if ch not in self.c2p:
                        self.c2p[ch] = []
                    self.c2p[ch].append(pid)
            for k, v in self.c2p.items():
                self.c2p[k] = sorted(v)

    def problem_c(self, problem, with_h):
        self.build_c2p_dict()
        remaining_chs = []
        probs = []
        for ch in problem.ch_list:
            if ch.startswith('C'):
                pids = [pid for pid in self.c2p[ch] if pid != problem.identifier]
                if len(pids) == 0:
                    remaining_chs.append(ch)
                else:
                    probs.append(random.choice(pids))
            else:
                if with_h:
                    remaining_chs.append(ch)
        probs_uniq = []
        for p in probs:
            if p not in probs_uniq:
                probs_uniq.append(p)
        probs = probs_uniq
        if len(probs) == 0:
            return self.direct_c(problem, with_h)
        prompt_lst = []
        if len(probs) == 1:
            prompt_lst.append(prompt_problem_single.format(
                sample_problem=self.show_solution(probs[0])))
        else:
            samples = '\n\n'.join([self.show_solution(prob) for prob in probs])
            prompt_lst.append(prompt_problem_multiple.format(
                sample_problems=samples))
        prompt_lst.append(prompt_problem_query.format(problem=problem))
        if len(remaining_chs) != 0:
            prompt_lst.append(prompt_problem_remaining.format(
                rem_ch_list_str=self.enumerate_ch(remaining_chs)))
        return sys_prompt, [''.join(prompt_lst)], []

    def build_c_cats_dict(self):
        try:
            self.c_cats
        except AttributeError:
            self.c_cats = dict()
            for cid, c in self.dataset.concepts.items():
                cat = c.category
                if cat not in self.c_cats:
                    self.c_cats[cat] = []
                self.c_cats[cat].append(cid)
            for c, v in self.c_cats.items():
                self.c_cats[c] = sorted(v)

    def misleading_c(self, problem, with_h):
        self.build_c_cats_dict()
        avoid_chs = set()
        for ch in problem.ch_list:
            if ch.startswith('C'):
                avoid_chs.update([c.identifier for c in self.dataset[ch].get_line()])
        new_chs = []
        for ch in problem.ch_list:
            if ch.startswith('H'):
                if with_h:
                    new_chs.append(ch)
            else:
                candidates = [c for c in self.c_cats[self.dataset[ch].category] if c not in avoid_chs]
                if len(candidates) == 0:
                    new_chs.append(ch)
                else:
                    new_chs.append(random.choice(candidates))
        return self.problem_with_ch_list(problem, new_chs)

    def partial_solution(self, problem, soln_frac):
        assert 0 <= soln_frac <= 1
        frac_num = (len(problem.solution.steps) - 1) * soln_frac
        whole = int(frac_num)
        if random.random() < (frac_num - whole):
            whole += 1
        if whole == 0:
            return self.problem_only(problem)
        soln_str = self.enumerate_solution(problem.solution.steps[:whole])
        prompt = prompt_partial_soln.format(
            problem=problem, 
            soln_list_str=soln_str)
        return sys_prompt, [prompt], []

    def few_shot(self, problem, num_shots):
        pids = [pid for pid in self.dataset.problems if pid != problem.identifier]
        pids = random.sample(pids, num_shots)
        prompts = []
        imputed_outputs = []
        for pid in pids:
            shot_prob = self.dataset.problems[pid]
            shot_input = prompt_main.format(problem=shot_prob)
            shot_output = 'Step-wise solution:\n'+self.enumerate_solution(shot_prob.solution.steps)
            prompts.append(shot_input)
            imputed_outputs.append(shot_output)
        p_input = prompt_main.format(problem=problem)
        prompts.append(p_input)
        return sys_prompt, prompts, imputed_outputs

    def get_all_prompts(self):
        return {'0-Shot': self.problem_only, 
                '5-Shot': partial(self.few_shot, num_shots=5), 
                '1/3 Soln': partial(self.partial_solution, soln_frac=1/3),
                '2/3 Soln': partial(self.partial_solution, soln_frac=2/3),
                'No C w/o H': partial(self.no_c, with_h=False), 
                'No C w/ H': partial(self.no_c, with_h=True), 
                'Direct C w/o H': partial(self.direct_c, with_h=False), 
                'Direct C w/ H': partial(self.direct_c, with_h=True), 
                'Name C w/o H': partial(self.name_c, with_h=False), 
                'Name C w/ H': partial(self.name_c, with_h=True), 
                'Example C w/o H': partial(self.example_c, with_h=False), 
                'Example C w/ H': partial(self.example_c, with_h=True), 
                'Root C w/o H': partial(self.root_c, with_h=False), 
                'Root C w/ H': partial(self.root_c, with_h=True), 
                'Problem C w/o H': partial(self.problem_c, with_h=False), 
                'Problem C w/ H': partial(self.problem_c, with_h=True), 
                'Misleading C w/o H': partial(self.misleading_c, with_h=False), 
                'Misleading C w/ H': partial(self.misleading_c, with_h=True)
               }

    def get_all_prompt_names(self):
        return list(self.get_all_prompts().keys())

    def construct_prompt(self, prompt_name, problem):
        prompts = self.get_all_prompts()
        if prompt_name not in prompts:
            raise ValueError(f'Invalid prompt name: {prompt_name}. ' + \
                f'The list of choices is: {list(prompts.keys())}')
        func = prompts[prompt_name]
        return func(problem)
