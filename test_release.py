import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch, Mock

from shotweave import design as d, measurements as m, applications as a
from shotweave.sequential import confidence_sequence, sequential
from shotweave.experiment import run_experiment, confidence_interval, summarize


class ReleaseTests(unittest.TestCase):
    def test_strategies(self):
        models = (m.Readout(0,0),m.Readout(.3,.3))
        for method in ("uniform","weighted","optimal","adaptive","adaptive-optimal","robust","bayesian","noise-aware"):
            result = run_experiment(108,[1,1],lambda ns:[n//2 for n in ns],method,4,readout=models)
            self.assertEqual(result.total_shots,108)
            lo,hi = confidence_interval(result,[1,1])
            self.assertLess(lo,1)
            self.assertGreater(hi,1)
            if method=="noise-aware":
                self.assertGreater(result.final_shots[1],2*result.final_shots[0])
        self.assertEqual(run_experiment(10,[1,1],lambda ns:ns,"robust",variance_bounds=[(0,0),(0,1)]).final_shots,(1,9))

    def test_domains(self):
        for fn,args in [
            (d.integer_cost,(5,[-1],[1])), (d.integer_cost,(5,[1],[0])),
            (d.integer_cost,(5,[1,2],[1])), (d.robust_allocation,(4,[1],[(1,0)])),
            (d.Resources,(1,-1,0,1)), (d.precision_plan,([1],0,4)),
            (d.precision_plan,([1],.01,4)), (d.Posterior,(0,1)),
            (d.Posterior(1,1).update,(3,2)), (d.bayesian_allocation,(4,[1],[])),
            (d.pilot_risks,(4,[1],[2],[1],None,2)), (d.pilot_risks,(4,[1],[.5],[5],None,2)),
            (m.compatible,("X","XZ")), (m.covariance,([[1]],)), (m.covariance,([[1],[2,3]],)),
            (m.group_variance,([1],[])), (m.Readout,(.5,.5)), (m.Readout(0,0).correct,(2,1)),
            (a.grouped_energy,(0,[],[])), (a.BackendRecord,("a","b","c","d",2,3,1)),
            (a.bornlens_rows,("x","x",[2],[3],[1])), (confidence_sequence,([],[],[])),
            (confidence_sequence,([1],[2],[3])), (sequential,([],None,10,2,2,.5)),
            (summarize,([float("nan")],[2],[1])),(run_experiment,(10,[1],None,"absent")),
            (run_experiment,(10,[0],None)),(run_experiment,(0,[1],None)),
            (run_experiment,(10,[1],None,"noise-aware")),
        ]:
            with self.subTest(fn=fn,args=args), self.assertRaises(ValueError):
                fn(*args)
        with self.assertRaises(ValueError):
            run_experiment(10,[1,1],None,"uniform",readout=[m.Readout(0,0)])
        with self.assertRaises(ValueError):
            run_experiment(10,[1],None,"weighted",variance_bounds=[(0,1)])
        self.assertEqual(a.bornlens_rows("x","y",[2],[3],[1])[-1]["total_shots"],5)
        self.assertEqual(d.robust_allocation(4,[1,1],[(0,0)]*2),[2,2])
        self.assertEqual(d.continuous_cost(7,[.001,10],[1,1]),(1.,6.))

    def test_cli(self):
        from shotweave.__main__ import main
        for payload in ['{"coefficients":[1],"shots":[4],"successes":[2]}','{}']:
            with patch("sys.argv",["shotweave"]), patch("sys.stdin",io.StringIO(payload)), redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
                if payload=="{}":
                    with self.assertRaises(SystemExit):
                        main()
                else:
                    main()
                    self.assertIn("0.5",out.getvalue())

    def test_sampler_errors(self):
        from shotweave.qiskit import SamplerAdapter
        circuit=Mock(num_parameters=0,cregs=[])
        for circuits in ([],[circuit]):
            with self.assertRaises(ValueError):
                SamplerAdapter(Mock(),circuits)
        register=Mock(size=1)
        register.name="meas"
        circuit.cregs=[register]
        sampler=Mock()
        adapter=SamplerAdapter(sampler,[circuit])
        with self.assertRaises(ValueError):
            adapter([])
        for results in ([],[Mock(data={"meas":Mock()})]):
            if results:
                results[0].data["meas"].get_counts.return_value={"2":1}
            sampler.run.return_value.result.return_value=results
            with self.assertRaises(ValueError):
                adapter([1])
