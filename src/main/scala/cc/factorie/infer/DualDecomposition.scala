package cc.factorie.infer

import cc.factorie.la._
import cc.factorie.variable._
import cc.factorie.model._
import cc.factorie.optimize.Example
import cc.factorie.util.DoubleAccumulator

/**
 * User: apassos
 * Date: 6/15/13
 * Time: 7:38 AM
 */

trait WarmStartWeightedSummary {
  def summary: Summary
  def infer(): Unit
  def incrementWeights(v: DiscreteVar, t: Tensor, d: Double)
}

case class ModelWithInference[M,V](vars: V, model: M)(implicit val infer: (V,M) => WarmStartWeightedSummary) {
  def summary = infer(vars, model)
}

class DualDecomposition(stepSize: (Int,Int) => Double = DualDecomposition.LearningRates.expDualIncrease(1.0, 0.9)) extends /*Infer[Seq[WarmStartWeightedSummary], Seq[(Int,DiscreteVar,Int,DiscreteVar)]] with*/ cc.factorie.util.GlobalLogging {
  def infer(summaries: Seq[WarmStartWeightedSummary], constraints: Seq[(Int, DiscreteVar, Int, DiscreteVar)]): MAPSummary = {
    var dual = summaries.map(_.summary.logZ).sum
    var updated = true
    var dualIncreases = 1
    var iterations = 1
    while (updated) {
      summaries.map(_.infer()) // summaries are responsible for caching inference results if they did not change
      val newDual = summaries.map(_.summary.logZ).sum
      if (newDual > dual) dualIncreases += 1
      dual = newDual
      logger.info(s"Dual: $dual}")
      updated = false
      for ((i1, v1, i2, v2) <- constraints) {
        val discreteMarginal1 = summaries(i1).summary.marginal(v1).asInstanceOf[DiscreteMarginal]
        val discreteMarginal2 = summaries(i2).summary.marginal(v2).asInstanceOf[DiscreteMarginal]
        if (discreteMarginal1.proportions.maxIndex != discreteMarginal2.proportions.maxIndex) {
          updated = true
          val t = new SparseIndexedTensor1(discreteMarginal1.proportions.length)
          val rate = 1.0/(1 + dualIncreases)
          t += (discreteMarginal1.proportions.maxIndex, -rate)
          t += (discreteMarginal2.proportions.maxIndex, rate)
          val step = stepSize(iterations, dualIncreases)
          summaries(i1).incrementWeights(v1, t, step)
          summaries(i2).incrementWeights(v2, t, -step)
        }
      }
      iterations += 1
    }
    val as = new HashMapAssignment(ignoreNonPresent=false)
    for (summary <- summaries) {
      implicit val d = new DiffList
      summary.summary.setToMaximize(d)
      for (diff <- d; v = diff.variable)
        as.update(v, v.value)
      d.undo()
    }
    new MAPSummary(as, summaries.flatMap(_.summary.factorMarginals.map(_.factor)).distinct)
  }
}

/**
 * This represents a linear constraint. It can compute the score of the constraint,
 * store a dual variable, and has factors scoring the constraint with the weight of
 * the dual variable.
 * @param variables The constrained variables.
 * @param tensors The constraint tensors, one per variable. The value of the constraint
 *                is the dot of each tensor with the corresponding variable.
 * @param lambda The value of the dual variable for this constraint.
 */
class LinearConstraintVariable(val variables: Seq[DiscreteVar], val tensors: Seq[Tensor1], var lambda: Double) {
  def score: Double = variables.zip(tensors).map({ case (v,t) => v.value dot t }).sum
  val factors = variables.zip(tensors).map({
    case (v,t) => new Factor1[DiscreteVar](v) {
      def score(v1: DiscreteVar#Value) = lambda * t.dot(v1)
    }
  })
}

/**
 * Base trait for dual decomposition constraints.
 */
trait DDConstraint {
  /**
   * @return the variable being constrained
   */
  def variable: LinearConstraintVariable
  /**
   * @return the increment this constraint has on the value of the dual objective.
   */
  def dualIncrement: Double

  /**
   * Updates the parameters to satisfy the constraint more.
   * @param lrate The current learning rate.
   * @return Whether the constraint needs another update.
   */
  def update(lrate: Double): Boolean
}

/**
 * Represents a linear equality constraint.
 * @param variable The variable being constrained
 * @param target Its target value
 */
class LinearEqualityConstraint(val variable: LinearConstraintVariable, val target: Double) extends DDConstraint {
  def dualIncrement: Double = - variable.lambda
  def update(lrate: Double): Boolean = {
    val v = variable.score
    if (v < target) {
      variable.lambda += lrate
      true
    } else if (v > target) {
      variable.lambda -= lrate
      true
    } else false
  }
}

/**
 * Represents a dual decomposition constraint that a variable has to be greater than or equal to a target.
 * @param variable the variable
 * @param target its target value
 */
class LinearGreaterThanConstraint(val variable: LinearConstraintVariable, val target: Double) extends DDConstraint {
  def dualIncrement: Double = -variable.lambda
  def update(lrate: Double): Boolean = {
    val v = variable.score
    if (v > target) {
      if (variable.lambda >= 0) {
        variable.lambda = math.max(variable.lambda-lrate, 0.0)
        true
      } else {
        false
      }
      variable.lambda = 0
      true
    } else if (v == target) {
      false
    } else {
      variable.lambda += lrate
      true
    }
  }
}

/**
 * Represents a linear term in the objective that penalizes a model if a linear greater than constraint is
 * not satisfied.
 * @param variable The variable
 * @param target Its target value
 * @param penalty The penalty for not satisfying the constraint.
 */
class LinearGreaterThanPenalty(val variable: LinearConstraintVariable, val target: Double, val penalty: Double) extends DDConstraint {
  def dualIncrement: Double = -variable.lambda
  def update(lrate: Double): Boolean = {
    val v = variable.score
    if (v > target) {
      if (variable.lambda >= 0) {
        variable.lambda = math.max(variable.lambda-lrate, 0.0)
        true
      } else {
        false
      }
      variable.lambda = 0
      true
    } else if (v == target) {
      false
    } else {
      variable.lambda = math.min(variable.lambda + lrate, penalty)
      true
    }
  }
}

class DDSummary(val base: Summary, val converged: Boolean, dual: Double) extends Summary {
  /** The collection of all Marginals available in this Summary */
  def marginals = base.marginals

  /** If this Summary has a univariate Marginal for variable v, return it; otherwise return null. */
  def marginal(v: Var) = base.marginal(v)

  /** If this Summary has a Marginal that touches all or a subset of the neighbors of this factor
      return the Marginal with the maximally-available subset. */
  def marginal(factor: Factor) = base.marginal(factor)

  def factorMarginals = base.factorMarginals

  def logZ = dual
}

/**
 * Base class for doing dual decomposition style inference with the above constraints.
 * @param variables The variables to run inference on
 * @param constraints The linear constraints
 * @param baseModel The base model
 * @param baseInfer The inference method, must be able to handle CombinedModels
 * @param baseLrate The initial learning rate
 * @param lRateExp Learning rate decrease exponent
 * @param maxIterations The maximum number of iterations
 */
class DualDecompositionInference(val variables: Iterable[DiscreteVar], val constraints: Seq[DDConstraint], val baseModel: Model, val baseInfer: Infer[Iterable[DiscreteVar],Model], val baseLrate: Double, val lRateExp: Double, val maxIterations: Int) {
  def infer: DDSummary = {
    var needToUpdate = true
    constraints.foreach(c => assert(c.variable.variables == variables))
    val ddModel = new ItemizedModel()
    constraints.foreach(c => ddModel ++= c.variable.factors)
    val combinedModel = new CombinedModel(ddModel, baseModel)
    var summary = baseInfer.infer(variables, combinedModel)
    summary.setToMaximize
    var dual = summary.logZ + constraints.map(_.dualIncrement).sum
    var oldDual = Double.PositiveInfinity
    var lrate = baseLrate
    var it = 0
    while (needToUpdate && it < maxIterations) {
      it += 1
      if (dual > oldDual) lrate *= lRateExp
      oldDual = dual
      needToUpdate = false
      for (c <- constraints) { if (c.update(lrate)) needToUpdate = true }
      if (needToUpdate) {
        summary = baseInfer.infer(variables, combinedModel)
        summary.setToMaximize
      }
      dual = summary.logZ + constraints.map(_.dualIncrement).sum
    }
    new DDSummary(summary, it < maxIterations, dual)
  }
}

abstract class ConstraintPenaltyTemplate(val parameters: cc.factorie.Parameters) {
  val weight = parameters.Weights(new DenseTensor1(1))

  // note: the constraint returned has to use weight.value(0) as its penalty or it won't work
  def getConstraint(variables: Iterable[DiscreteVar]): LinearGreaterThanPenalty
}

class DualDecompositionPerceptronExample(val model: Model with Parameters, val penaltyTemplates: Seq[ConstraintPenaltyTemplate], val constraints: Seq[DDConstraint], val variables: Iterable[DiscreteVar], baseInfer: Infer[Iterable[DiscreteVar],Model]) extends Example {
  penaltyTemplates.foreach(t => assert(t.parameters eq model.parameters))
  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator) = {
    val templateConstraints = penaltyTemplates.map(_.getConstraint(variables))
    val actualConstraints = constraints ++ templateConstraints
    val summary = new DualDecompositionInference(variables, actualConstraints, model, baseInfer, 1.0, 0.9, 100).infer
    if (value ne null) value.accumulate(-summary.logZ)
    for (factorMarginal <- summary.factorMarginals) {
      factorMarginal.factor match {
        case factor: DotFamily#Factor if factor.family.isInstanceOf[DotFamily] =>
          val aStat = factor.assignmentStatistics(TargetAssignment)
          if (value != null) value.accumulate(factor.statisticsScore(aStat))
          if (gradient != null) {
            gradient.accumulate(factor.family.weights, aStat)
            gradient.accumulate(factor.family.weights, factorMarginal.tensorStatistics, -1.0)
          }
        case factor: Family#Factor if !factor.family.isInstanceOf[DotFamily] =>
          if (value != null) value.accumulate(factor.assignmentScore(TargetAssignment))
      }
    }
    val inferredValues = templateConstraints.map(_.variable.score)
    variables.foreach({ case v: LabeledMutableDiscreteVarWithTarget => v.setToTarget(null)})
    val targetValues = templateConstraints.map(_.variable.score)
    for (i <- 0 until templateConstraints.length) {
      if (inferredValues(i) > targetValues(i)) gradient.accumulate(penaltyTemplates(i).weight, new SingletonTensor1(1, 0, -1))
      else if (inferredValues(i) < targetValues(i)) gradient.accumulate(penaltyTemplates(i).weight, new SingletonTensor1(1, 0, 1))
    }
  }
}


object InferByDualDecomposition extends DualDecomposition

object DualDecomposition {
  class WeightedSummaryWithBP(vars: Iterable[DiscreteVar], model: Model, baseInfer: MaximizeByBP) extends WarmStartWeightedSummary {
    val weightedModel = new ItemizedModel
    val combinedModel = new CombinedModel(model, weightedModel)
    var summary = baseInfer.infer(vars, combinedModel)
    def infer() { summary = baseInfer.infer(vars, combinedModel)}
    case class WeightedFactor(var v: DiscreteVar, var t: Tensor, var d: Double) extends Factor1[DiscreteVar](v) {
      def score(v1: DiscreteVar#Value) = d*(v1 dot t)
      override def valuesScore(v1: Tensor) = d*(v1 dot t)
      override def equals(other: Any) = this eq other.asInstanceOf[AnyRef]
    }
    val factors = collection.mutable.ArrayBuffer[WeightedFactor]()
    def incrementWeights(v: DiscreteVar, t: Tensor, d: Double) {
      factors += new WeightedFactor(v, t, d)
      weightedModel += factors.last
    }
  }
  def getBPInferChain(vars: Iterable[DiscreteVar], model: Model) = new WeightedSummaryWithBP(vars, model, MaximizeByBPChain)
  def getBPInferTree(vars: Iterable[DiscreteVar], model: Model) = new WeightedSummaryWithBP(vars, model, MaximizeByBPTree)
  def getBPInferLoopy(vars: Iterable[DiscreteVar], model: Model) = new WeightedSummaryWithBP(vars, model, MaximizeByBPLoopy)

  object LearningRates {
    def expDualIncrease(eta: Double, c: Double) = (iteration: Int, dualIncreases: Int) => eta*math.pow(c, -dualIncreases)
    def expT(eta: Double, c: Double) = (iteration: Int, dualIncreases: Int) => eta*math.pow(c, -iteration)

    def invDualIncrease(eta: Double) = (iteration: Int, dualIncreases: Int) => eta/dualIncreases
    def invT(eta: Double) = (iteration: Int, dualIncreases: Int) => eta/iteration

    def invSqrtDualIncrease(eta: Double) = (iteration: Int, dualIncreases: Int) => eta/math.sqrt(dualIncreases)
    def invSqrtT(eta: Double) = (iteration: Int, dualIncreases: Int) => eta/math.sqrt(iteration)
  }
}

