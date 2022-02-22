# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
import fnmatch
import weakref
from functools import partial


class StateMachine:
    def __init__(self, name, handler, **kwargs):
        self.transit_map_ = {}
        self.name_ = name
        self.current_state_ = None
        self.handler_ = handler
        self.handler_.parent = weakref.ref(self)
        self.events_ = set()
        self.event = None
        self.__dict__.update(kwargs)
        self.next_state = None
        self.default_handle = None
        if 'start' in kwargs:
            self.current_state_ = kwargs['start']
        self.parser_ = re.compile(
            r'^([^ \t\n\r\f\v\->]*)[\s]*[\-]+[>]?[\s]*([^ \t\n\r\f\v\->]*)[\s]*[\-]+>[\s]*([^ \t\n\r\f\v\->]*)$')
        cls = handler.__class__
        for k, v in cls.__dict__.items():
            if hasattr(v, '__call__'):
                if v.__doc__ is not None:
                    self._add_transit_by(v, v.__doc__)
                else:
                    if callable(v):
                        self.__dict__[k] = partial(v, self.handler_)

    def _event_func(self, *args, **kwargs):
        self.handle_event(self.event, *args, **kwargs)

    def _add_transit_by(self, v, trans):
        for tran in trans.split('\n'):
            tran = tran.strip()
            trans_line = self.parser_.match(tran)
            if trans_line:
                self.add_transit(trans_line.group(1), trans_line.group(2),
                                 trans_line.group(3), v)
                if self.current_state_ is None:
                    self.current_state_ = trans_line.group(1)
                self.events_.add(trans_line.group(2))
            elif tran.strip() == 'default_handle':
                self.default_handle = v

    def __getattr__(self, item):
        for event in self.events_:
            if fnmatch.fnmatch(item, event):
                self.event = item
                return self._event_func
        if item in self.__dict__:
            return self.__dict__[item]
        if item in self.handler_.__dict__:
            return self.handler_.__dict__[item]
        return None

    def add_transit(self, s0, e, s1, func=None):
        if s0 in self.transit_map_:
            handles = self.transit_map_[s0]
            handles[e] = {'func': func, 'state': s1}
        else:
            self.transit_map_[s0] = {e: {'func': func, 'state': s1}}

    def start_state(self, s):
        self.current_state_ = s

    def handle_event(self, e, *args, **kwargs):
        handled = False
        self.handler_.current_event = e
        if self.current_state_ in self.transit_map_:
            handles = self.transit_map_[self.current_state_]
            for k, trans in handles.items():
                if fnmatch.fnmatch(e, k):
                    func = trans['func']
                    self.next_state = handles[k]['state']
                    ret = func(self.handler_, *args, **kwargs)
                    current_state = self.current_state_
                    transit_done = True
                    if ret is None:
                        self.current_state_ = self.next_state
                    elif ret == True:
                        self.current_state_ = self.next_state
                    else:
                        transit_done = False
                    handled = True
                    if self.debug:
                        if transit_done:
                            print("[%s][%s -- %s --> %s]" % (self.name_,
                                                             current_state,
                                                             e,
                                                             self.current_state_))
                        else:
                            print("[%s][%s -- %s --> %s[%s]][Transition is refused]" % (self.name_,
                                                                                        current_state,
                                                                                        e,
                                                                                        self.current_state_,
                                                                                        self.next_state))
                        # for a in args:
                        #     print(a)
                        # for k, v in kwargs.items():
                        #     print('%s=%o' %(k,v))
        if not handled:
            if self.debug:
                print("[%s][%s -- %s <-- %s]" % (self.name_,
                                                 self.current_state_,
                                                 e,
                                                 'not handled'))
            if self.default_handle:
                self.default_handle(self.handler_, *args, **kwargs)

    def get_state(self):
        return self.current_state_

    def set_next_state(self, next_state):
        self.next_state = next_state

    def dump(self):
        for (s, v) in self.transit_map_.items():
            print(s, v)


chapters = [
    ("Miscellaneous constants",   0, False),
    ("Error classes and codes",   0, False),
    ("New datatypes",             1, True),
    ("C datatypes",               0, True),
    ("MPI predefined handles",    0, True),
    ("NULL handles",              0, True),
    ("MPIT Verbosity Levels",     0, False),
    ("MPIT Scopes",               0, False),
    ("MPIT Object Binding",       0, False),
    ("MPIT pvar classes",         0, False),
    ("MPI_Init_thread constants", 0, False),
    ("More constants",            0, False)
]

CURRENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

MPI_H_IN = os.path.join(CURRENT_PATH, "third_party", "ompi-3.1.6", "ompi", "include", "mpi.h.in")
MPI_H_IMPL = os.path.join(CURRENT_PATH, "include", "nbla", "cuda", "communicator", "dl_mpi.h.tmpl")
DL_MPI_H = os.path.join(CURRENT_PATH, "include", "nbla", "cuda", "communicator", "dl_mpi.h")


class DefineExtractor:
    def __init__(self):
        self.found = False
        self.current = None
        self.need_mod = False
        self.defines = {}
        self.chapters = {}
        n = 1000
        for comment, times, need_mod in chapters:
            if n > len(comment):
                n = len(comment)
        for comment, times, need_mod in chapters:
            self.chapters[comment[:n]] = (comment, times, need_mod)
        self.match_len = n

    def check_if_meet_block(self, line):
        self.current = None
        k = line.replace("/*", "").replace(
            "*/", ""
        ).replace(
            "*", ""
        ).strip()[:self.match_len]

        if k in self.chapters:
            count = self.chapters[k][1]
            if count != 0:
                count -= 1
                self.chapters[k] = (self.chapters[k][0], count, self.chapters[k][2])
            else:
                nm = self.chapters[k][0]
                chapter_name = nm.lower().replace(" ", "_")
                print(f"found {chapter_name}")
                self.current = chapter_name
                self.need_mod = self.chapters[k][2]
                self.defines[chapter_name] = []
                self.found = True
                count -= 1
                self.chapters[k] = (self.chapters[k][0], count, self.chapters[k][2])


    def feed_single_comment(self, line):
        '''idle -- single_comment --> idle
        '''
        self.check_if_meet_block(line)

    def feed_common_line(self, line):
        '''idle -- common_line --> idle
        '''
        if self.current:
            line = line.strip()
            if line != '':
                if self.need_mod:
                    line = line.replace("ompi_", "_ompi_")
                self.defines[self.current].append(line)
                if self.current == 'error_classes_and_codes':
                    print(line)

    def comment_begin(self):
        '''idle -- comment_begin --> comment_block
        '''
        pass

    def comment_block(self, line):
        '''comment_block -- comment_line --> comment_block
        '''
        if not self.found:
            self.check_if_meet_block(line)

    def comment_end(self):
        '''comment_block -- comment_end --> idle
        '''
        self.found = False

    def render(self):
        from mako.template import Template
        from mako import exceptions

        outputs = {}
        for k, v in self.defines.items():
            outputs[k] = "\n".join(v)

        tmpl = Template(filename=MPI_H_IMPL)
        try:
            return tmpl.render(**outputs)
        except Exception as e:
            import sys
            print('-' * 78, file=sys.stderr)
            print('Template exceptions', file=sys.stderr)
            print('-' * 78, file=sys.stderr)
            print(exceptions.text_error_template().render(), file=sys.stderr)
            print('-' * 78, file=sys.stderr)
            raise e


extractor = StateMachine('DefineExtractor', DefineExtractor(), start='idle', debug=False)

with open(MPI_H_IN, "r") as f:
    for line in f.readlines():
        line = line.strip()
        if line.startswith("/*") and line.endswith("*/"):
            extractor.single_comment(line)
        elif line.startswith("/*"):
            extractor.comment_begin()
        elif line.startswith("*") and line.endswith("*/"):
            extractor.comment_end()
        elif line.startswith("*"):
            extractor.comment_line(line)
        else:
            extractor.common_line(line)

with open(DL_MPI_H, "w") as f:
    f.write(extractor.render())
