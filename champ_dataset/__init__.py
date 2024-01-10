
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, asdict
from typing import List, Dict
from natsort import natsorted

import os
try:
    from importlib import resources as impresources
    BASE_FOLDER = impresources.files(__name__)
except:
    BASE_FOLDER = os.path.split(__file__)[0]

@dataclass
class Conversation():
    messages: List[Message] = None

    @classmethod
    def from_dict(cls, d):
        if 'messages' in d:
            messages = [Message.from_dict(e) for e in d['messages']]
        else:
            messages = None
        return cls(messages=messages)

    def validate(self):
        assert isinstance(self.messages, list) and \
            all(isinstance(message, Message) for message in self.messages) and \
            all(message.validate() for message in self.messages)
        return True

@dataclass
class Message():
    role: str = None
    text: str = None
    error: str = None

    @classmethod
    def from_dict(cls, d):
        m = cls(role=d.get('role', None), text=d.get('text', None), error=d.get('error', None))
        return m

    def validate(self):
        assert self.role in ['System', 'User', 'Imputation'] or self.role.startswith('Model.')
        assert (isinstance(self.text, str) and self.error in [None, 'LENGTH_CURRENT']) or \
               (self.text is None and self.error == 'LENGTH_PREVIOUS')
        if self.error is not None:
            assert self.role.startswith('Model.')
        return True

@dataclass
class FWSAnnotation():
    text: str = None
    start_idx: int = None
    end_idx: int = None
    author: str = None

    @classmethod
    def from_dict(cls, d):
        return cls(text=d.get('text', None), start_idx=d.get('start_idx', None), 
                   end_idx=d.get('end_idx', None), author=d.get('author', None))

    def wrong_step(self):
        if self.start_idx is None:
            return 'No mistake'
        else:
            return self.text[self.start_idx : self.end_idx]

    def validate(self):
        assert isinstance(self.text, str)
        assert self.start_idx is None or isinstance(self.start_idx, int)
        assert self.end_idx is None or isinstance(self.end_idx, int)
        assert (self.start_idx is None) == (self.end_idx is None)
        assert isinstance(self.author, str)
        return True

@dataclass
class Concept():
    identifier: str = None
    name: str = None
    _raw_text: str = None
    _text: str = None
    category: str = None
    parent: Concept = None

    @classmethod
    def from_dict(cls, d):
        if d is None:
            return None
        parent = cls.from_dict(d['parent']) if 'parent' in d else None
        c = cls(identifier=d.get('identifier', None), name=d.get('name', None), 
                   _raw_text=d.get('_raw_text', None), category=d.get('category', None), 
                   parent=parent)
        if c._raw_text is not None:
            c.raw_text = c._raw_text
        return c

    @property
    def raw_text(self):
        return self._raw_text

    @raw_text.setter
    def raw_text(self, v):
        assert v.count('@@') % 2 == 0, f'Incorrect text format: {v}'
        self._raw_text = v
        self._text = v.replace('@@', '')

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, v):
        raise NotImplementedError('"text" should not be set directly. Set "raw_text" instead.')

    def get_root(self):
        return self.get_line()[-1]

    def get_line(self):
        cur = self
        line = [cur]
        while cur.parent is not None:
            cur = cur.parent
            line.append(cur)
        return line

    def validate(self):
        assert isinstance(self.identifier, str) and self.identifier.startswith('C_')
        assert isinstance(self.text, str)
        assert isinstance(self.category, str)
        assert self.name is None or isinstance(self.name, str)
        assert self.parent is None or (isinstance(self.parent, Concept) and self.parent.validate())
        return True

@dataclass
class Hint():
    identifier: str = None
    _raw_text: str = None
    _text: str = None

    @classmethod
    def from_dict(cls, d):
        h = cls(identifier=d.get('identifier', None), _raw_text=d.get('_raw_text', None))
        if h._raw_text is not None:
            h.raw_text = h._raw_text
        return h

    @property
    def raw_text(self):
        return self._raw_text

    @raw_text.setter
    def raw_text(self, v):
        assert v.count('@@') % 2 == 0, f'Incorrect text format: {v}'
        self._raw_text = v
        self._text = v.replace('@@', '')

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, v):
        raise NotImplementedError('"text" should not be set directly. Set "raw_text" instead.')

    def validate(self):
        assert isinstance(self.identifier, str) and self.identifier.startswith('H_')
        assert isinstance(self.text, str)
        return True

@dataclass
class SolutionStep():
    _raw_text: str = None
    _text: str = None
    ch_idxs: List[int] = None

    @classmethod
    def from_dict(cls, d):
        st = cls(_raw_text=d.get('_raw_text', None), ch_idxs=d.get('ch_idxs', None))
        if st._raw_text is not None:
            st.raw_text = st._raw_text
        return st

    @property
    def raw_text(self):
        return self._raw_text

    @raw_text.setter
    def raw_text(self, v):
        assert v.count('@@') % 2 == 0, f'Incorrect text format: {v}'
        self._raw_text = v
        self._text = v.replace('@@', '')

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, v):
        raise NotImplementedError('"text" should not be set directly. Set "raw_text" instead.')

    def validate(self):
        assert isinstance(self.text, str)
        assert isinstance(self.ch_idxs, list) and all(isinstance(idx, int) for idx in self.ch_idxs)
        return True

@dataclass
class Solution():
    steps: List[SolutionStep] = None

    @classmethod
    def from_dict(cls, d):
        if 'steps' in d:
            steps = [SolutionStep.from_dict(e) for e in d['steps']]
        else:
            steps = None
        return cls(steps=steps)

    def validate(self):
        assert isinstance(self.steps, list) and \
        all(isinstance(step, SolutionStep) for step in self.steps) and \
        all(step.validate() for step in self.steps)
        return True

@dataclass
class Problem():
    identifier: str = None
    _raw_text: str = None
    _text: str = None
    category: str = None
    solution: Solution = None
    _raw_answer: str = None
    _answer: str = None
    ch_list: List[str] = None
    fws_annotations: Dict[str, FWSAnnotation] = None
    conversations: Dict[str, Conversation] = None

    @classmethod
    def from_dict(cls, d):
        solution = Solution.from_dict(d['solution']) if 'solution' in d else None
        if 'conversations' in d and d['conversations'] is not None:
            conversations = dict()
            for k, v in d['conversations'].items():
                conversations[k] = Conversation.from_dict(v)
        else:
            conversations = None
        if 'fws_annotations' in d and d['fws_annotations'] is not None:
            fws_annotations = dict()
            for k, v in d['fws_annotations'].items():
                fws_annotations[k] = FWSAnnotation.from_dict(v)
        else:
            fws_annotations = None
        p = cls(identifier=d.get('identifier', None), _raw_text=d.get('_raw_text', None), 
                category=d.get('category', None), solution=solution, 
                _raw_answer=d.get('_raw_answer', None), ch_list=d.get('ch_list', None), 
                fws_annotations=fws_annotations, conversations=conversations)
        if p._raw_text is not None:
            p.raw_text = p._raw_text
        if p._raw_answer is not None:
            p.raw_answer = p._raw_answer
        return p

    @property
    def raw_text(self):
        return self._raw_text

    @raw_text.setter
    def raw_text(self, v):
        assert v.count('@@') % 2 == 0, f'Incorrect text format: {v}'
        self._raw_text = v
        self._text = v.replace('@@', '')

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, v):
        raise NotImplementedError('"text" should not be set directly. Set "raw_text" instead.')

    @property
    def raw_answer(self):
        return self._raw_answer

    @raw_answer.setter
    def raw_answer(self, v):
        assert v.count('@@') % 2 == 0, f'Incorrect answer format: {v}'
        self._raw_answer = v
        self._answer = v.replace('@@', '')

    @property
    def answer(self):
        return self._answer

    @answer.setter
    def answer(self, v):
        raise NotImplementedError('"answer" should not be set directly. Set "raw_answer" instead.')

    def unique_ch_list(self):
        s = set()
        uniq_list = []
        for e in self.ch_list:
            if e.identifier not in s:
                s.add(e.identifier)
                uniq_list.append(e)
        return uniq_list

    def validate(self):
        assert isinstance(self.identifier, str) and self.identifier.startswith('P_')
        assert isinstance(self.text, str)
        assert isinstance(self.category, str)
        assert isinstance(self.solution, Solution) and self.solution.validate()
        assert isinstance(self.answer, str)
        assert isinstance(self.ch_list, list) and all(isinstance(ch, str) for ch in self.ch_list)
        assert isinstance(self.fws_annotations, dict) and \
               all(isinstance(k, str) for k in self.fws_annotations.keys()) and \
               all(isinstance(v, FWSAnnotation) for v in self.fws_annotations.values()) and \
               all(v.validate() for v in self.fws_annotations.values())
        assert isinstance(self.conversations, dict) and \
               all(isinstance(k, str) for k in self.conversations.keys()) and \
               all(k.count('|')==1 for k in self.conversations.keys()) and \
               all(isinstance(v, Conversation) for v in self.conversations.values()) and \
               all(v.validate() for v in self.conversations.values())
        return True

@dataclass
class Dataset():
    problems: Dict[str, Problem] = None
    concepts: Dict[str, Concept] = None
    hints: Dict[str, Hint] = None

    @classmethod
    def from_json(cls, fn):
        with open(fn) as f:
            d = json.load(f)
        dataset = cls.from_dict(d)
        try:
            assert dataset.validate()
        except Exception as e:
            traceback.print_exc()
            print('Dataset validation fails')
        return dataset

    @classmethod
    def from_dict(cls, d):
        if 'problems' in d:
            problems = dict()
            for k, v in d['problems'].items():
                problems[k] = Problem.from_dict(v)
        else:
            problems = None
        if 'concepts' in d:
            concepts = dict()
            for k, v in d['concepts'].items():
                concepts[k] = Concept.from_dict(v)
        else:
            concepts = None
        if 'hints' in d:
            hints = dict()
            for k, v in d['hints'].items():
                hints[k] = Hint.from_dict(v)
        else:
            hints = None
        return cls(problems=problems, concepts=concepts, hints=hints)

    def __getitem__(self, identifier):
        if identifier[0] == 'P':
            return self.problems[identifier]
        elif identifier[0] == 'C':
            return self.concepts[identifier]
        elif identifier[0] == 'H':
            return self.hints[identifier]
        else:
            raise Exception(f'Unrecognized identifier {identifier}')

    def prune_ch(self, verbose=False):
        used_chs = set()
        for p in self.problems.values():
            for ch in p.ch_list:
                used_chs.add(ch.identifier)
        if verbose:
            print((f'We have {len(self.concepts)} concepts and {len(self.hints)} hints'))
            k_pruned = [k for k, v in self.concepts.items() if k not in used_chs]
            h_pruned = [k for k, v in self.hints.items() if k not in used_chs]
            print('Pruning concepts:', k_pruned)
            print('Pruning hints:', h_pruned)
        self.concepts = {k: v for k, v in self.concepts.items() if k in used_chs}
        self.hints = {k: v for k, v in self.hints.items() if k in used_chs}
        if verbose:
            print((f'We now have {len(self.concepts)} concepts and {len(self.hints)} hints'))

    def to_json(self, fn):
        with open(fn, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    def validate(self):
        for p in self.problems.values():
            p.validate()
        for c in self.concepts.values():
            c.validate()
        for h in self.hints.values():
            h.validate()
        return True

    def sort(self):
        self.problems = {k: self.problems[k] for k in natsorted(self.problems.keys())}
        self.concepts = {k: self.concepts[k] for k in natsorted(self.concepts.keys())}
        self.hints = {k: self.hints[k] for k in natsorted(self.hints.keys())}

def load(fn):
    if not fn.endswith('.json'):
        fn = os.path.join(BASE_FOLDER, f'dataset_files/{fn}.json')
    with open(fn) as f:
        data = json.load(f)
    dataset = Dataset.from_dict(data)
    return dataset

from .prompt import PromptGenerator
