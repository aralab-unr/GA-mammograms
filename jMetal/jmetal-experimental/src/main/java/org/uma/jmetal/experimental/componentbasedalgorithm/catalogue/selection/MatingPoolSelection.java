package org.uma.jmetal.experimental.componentbasedalgorithm.catalogue.selection;

import java.util.List;
import org.uma.jmetal.solution.Solution;

@FunctionalInterface
public interface MatingPoolSelection<S extends Solution<?>> {
  List<S> select(List<S> solutionList) ;
}
