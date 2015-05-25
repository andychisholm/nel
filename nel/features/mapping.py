import math
import numpy

class FeatureMapper(object):
    def __init__(self, features):
        self.features = features
    
    def map(self, fv):
        raise NotImplementedError

    def __call__(self, doc):
        for chain in doc.chains:
            for c in chain.candidates:
                c.fv = self.map(numpy.array([c.features[f] for f in self.features]))
        return doc

    def feature_vector_length(self):
        raise NotImplementedError

class ZeroMeanUnitVarianceMapper(FeatureMapper):
    def __init__(self, features, means, stds):
        super(ZeroMeanUnitVarianceMapper,self).__init__(features)
        self.mean = means
        self.std = stds

    def map(self, fv):
        return (fv - self.mean) / self.std

    def feature_vector_length(self):
        return len(self.features)

class PolynomialMapper(ZeroMeanUnitVarianceMapper): 
    def __init__(self, features, means, stds):
        super(PolynomialMapper,self).__init__(features, means, stds)

    def map(self, fv):
        fv = list(super(PolynomialMapper, self).map(fv))
        
        sz = len(fv)
        for i in xrange(0, sz):
            for j in xrange(i, sz):
                weight = 1.0 if i != j else math.sqrt(2.0)
                fv.append(weight * fv[i]*fv[j])
        
        return numpy.array(fv)

    def feature_vector_length(self):
        n = len(self.features)
        return n + n*(n+1)/2

FEATURE_MAPPERS = {cls.__name__:cls for cls in [ZeroMeanUnitVarianceMapper,PolynomialMapper]}
