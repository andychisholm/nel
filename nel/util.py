import six
import Queue
import socket
import multiprocessing

from time import time
from itertools import chain
from collections import defaultdict
from bisect import bisect_left, bisect_right
from contextlib import contextmanager

from nel import logging
log = logging.getLogger()

def get_from_module(cid, mod_params, mod_name, instantiate=False, kwargs=None):
    if isinstance(cid, six.string_types):
        res = mod_params.get(cid)
        if not res:
            raise Exception('Invalid ' + str(mod_name) + ': ' + str(cid))
        if instantiate and not kwargs:
            return res()
        elif instantiate and kwargs:
            return res(**kwargs)
        else:
            return res
    return cid

def byte_to_char_map(byte_str, encoding='utf-8'):
    mapping = {}
    char_str = byte_str.decode(encoding)
    byte_offset, char_offset = 0, 0
    for char_offset, c in enumerate(char_str):
        mapping[byte_offset] = char_offset
        byte_offset += len(c.encode(encoding))
    mapping[byte_offset] = char_offset
    return mapping

def group(iteration, key_getter, value_getter):
    d = defaultdict(list)
    for item in iteration:
        d[key_getter(item)].append(value_getter(item))
    return d

def invert_grouping(g):
    d = defaultdict(list)
    for k, items in g.iteritems():
        for i in items:
            d[i].append(k)
    return d

def spanset_insert(indicies, begin, end):
    """ Determines if a span from an index set is occupied in O(log(n)) """
    b_idx = bisect_right(indicies, begin)
    e_idx = bisect_left(indicies, end)

    can_insert = b_idx == e_idx and \
                 (b_idx == 0 or indicies[b_idx - 1] != begin) and \
                 (e_idx == len(indicies) or indicies[e_idx] != end) and \
                 b_idx % 2 == 0

    if can_insert:
        indicies.insert(b_idx, begin)
        indicies.insert(b_idx + 1, end)

    return can_insert

def spawn_worker(f):
    def fun(wid, q_in, q_out, recycle_interval):
        job_count = 0
        while True:
            i,x = q_in.get()
            if i is None:
                break
            try:
                recycle_id = wid if job_count + 1 == recycle_interval else None

                q_out.put(((i, f(x)), recycle_id))
                job_count += 1

                if recycle_id != None:
                    return
            except Exception as e:
                log.error("Worker function exception: %s" % e)
                raise
    return fun

def iter_to_input_queue(iteration, q_in, p_control):
    iteration_len = 0
    for i, x in enumerate(iteration):
        q_in.put((i, x))
        iteration_len += 1

    p_control.send(iteration_len)
    p_control.close()

class parmapper(object):
    def __init__(self, job, nprocs = None, recycle_interval = 5):
        if nprocs == None:
            nprocs = multiprocessing.cpu_count() - 1
        self.job = job
        self.q_in = multiprocessing.Queue(1)
        self.q_out = multiprocessing.Queue(nprocs)
        self.recycle_interval = recycle_interval
        self.procs = [self.get_process(i) for i in range(nprocs)]

    def get_process(self, idx):
        return multiprocessing.Process(
            target=spawn_worker(self.job),
            args=(idx, self.q_in, self.q_out, self.recycle_interval))

    def run_process(self, idx):
        self.procs[idx].daemon = True
        self.procs[idx].start()

    def __enter__(self):
        for i in xrange(len(self.procs)):
            self.run_process(i)
        return self

    def recycle_worker(self, wid):
        worker = self.procs[wid]
        #log.debug('Recycling worker id=%i, pid=%i...' % (wid, worker.pid))
        worker.join()
        self.procs[wid] = self.get_process(wid)
        self.run_process(wid)

    def consume(self, producer):
        worker_pipe, control_pipe = multiprocessing.Pipe(True)
        async_input_iterator = multiprocessing.Process(target=iter_to_input_queue,args=(producer, self.q_in, worker_pipe))
        async_input_iterator.daemon = True
        async_input_iterator.start()

        expected_output_count = None
        output_count = 0

        while expected_output_count == None or expected_output_count > output_count:
            if expected_output_count == None and control_pipe.poll():
                expected_output_count = control_pipe.recv()
                #log.debug('Producer exhausted with %i items total, %i remaining...' % (expected_output_count, expected_output_count - output_count))
            try:
                # couldn't get this working without a busy wait
                out, recycle_wid = self.q_out.get_nowait()
                while True:
                    if recycle_wid != None:
                        self.recycle_worker(recycle_wid)
                    yield out
                    output_count += 1
                    out, recycle_wid = self.q_out.get_nowait()
            except Queue.Empty: pass

        async_input_iterator.join()

    def __exit__(self, t, value, traceback):
        for _ in self.procs:
            self.q_in.put((None,None))
        for p in self.procs:
            p.join() # todo: kill after some timeout

@contextmanager
def tcp_socket(host,port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host,port))
        yield s
    finally:
        try:
            s.shutdown(socket.SHUT_RDWR)
        except socket.error:
            pass
        except OSError:
            pass
        finally:
            s.close()

class trie(object):
    def __init__(self):
        self.Children = defaultdict(trie)
        self.Matches = set()

    def insert_many(self, sequence, entities):
        if len(entities) > 0:
            self._insert(sequence, entities, 0, True)

    def insert(self, sequence, e):
        self._insert(sequence, e, 0, False)

    def _insert(self, sequence, e, offset, multi):
        if offset < len(sequence):
            item = sequence[offset]

            self.Children[item]._insert(sequence, e, offset + 1, multi)
        else:
            if multi:
                for entity in e:
                    self.Matches.add((entity, offset))
            else:
                self.Matches.add((e, offset))

    def iter_matches(self):
        for e in self.Matches: yield e

    def scan(self, seq):
        for i in xrange(0, len(seq)):
            for m in self.match(seq, i, True, True):
                yield m

    def match(self, seq, offset = 0, subsequences = False, inorder = True):
        # if we are yielding subsequence matches, or the sequence
        # is complete return all entities for the current node
        current = [(e, (offset - length, offset)) for e, length in self.iter_matches()] if subsequences or offset == len(seq) else None

        # iteration for the next items in the sequence
        pending = None
        if seq and offset < len(seq):
            token = seq[offset]
            if token in self.Children:
                pending = self.Children[token].match(seq, offset + 1, subsequences, inorder)

        if current and pending:
            return chain(current, pending) if inorder else chain(pending, current)
        
        return current or pending or []
