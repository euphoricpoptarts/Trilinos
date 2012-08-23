// @HEADER
//
// ***********************************************************************
//
//   Zoltan2: A package of combinatorial algorithms for scientific computing
//                  Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Karen Devine      (kddevin@sandia.gov)
//                    Erik Boman        (egboman@sandia.gov)
//                    Siva Rajamanickam (srajama@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_XpetraCrsMatrixInput.hpp>
#include <Zoltan2_XpetraVectorInput.hpp>
#include <Zoltan2_TestHelpers.hpp>
#include <iostream>
#include <limits>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_Vector.hpp>
#include <MatrixMarket_Tpetra.hpp>

//#include <Zoltan2_Memory.hpp>  KDD User app wouldn't include our memory mgr.

using Teuchos::RCP;
using namespace std;

/////////////////////////////////////////////////////////////////////////////
// Program to demonstrate use of Zoltan2 to partition a TPetra matrix 
// (read from a MatrixMarket file or generated by MueLuGallery).
// Usage:
//     a.out [--inputFile=filename] [--outputFile=outfile] [--verbose] 
//           [--x=#] [--y=#] [--z=#] [--matrix={Laplace1D,Laplace2D,Laplace3D}
// Karen Devine, 2011
/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// Eventually want to use Teuchos unit tests to vary z2TestLO and
// GO.  For now, we set them at compile time based on whether Tpetra
// is built with explicit instantiation on.  (in Zoltan2_TestHelpers.hpp)

typedef lno_t z2TestLO;
typedef gno_t z2TestGO;
typedef scalar_t Scalar;

typedef Kokkos::DefaultNode::DefaultNodeType Node;
typedef Tpetra::CrsMatrix<Scalar, z2TestLO, z2TestGO> SparseMatrix;
typedef Tpetra::Vector<Scalar, z2TestLO, z2TestGO> Vector;

typedef Zoltan2::XpetraCrsMatrixInput<SparseMatrix> SparseMatrixAdapter;
typedef Zoltan2::XpetraVectorInput<Vector> VectorAdapter;

#define epsilon 0.00000001

/////////////////////////////////////////////////////////////////////////////
int main(int narg, char** arg)
{
  std::string inputFile = "";            // Matrix Market file to read
  std::string outputFile = "";           // Matrix Market file to write
  bool verbose = false;                  // Verbosity of output
  int testReturn = 0;

  ////// Establish session.
  Teuchos::GlobalMPISession mpiSession(&narg, &arg, NULL);
  RCP<const Teuchos::Comm<int> > comm =
    Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  int me = comm->getRank();

  // Read run-time options.
  Teuchos::CommandLineProcessor cmdp (false, false);
  cmdp.setOption("inputFile", &inputFile,
                 "Name of the Matrix Market sparse matrix file to read; "
                 "if not specified, a matrix will be generated by MueLu.");
  cmdp.setOption("outputFile", &outputFile,
                 "Name of the Matrix Market sparse matrix file to write, "
                 "echoing the input/generated matrix.");
  cmdp.setOption("verbose", "quiet", &verbose,
                 "Print messages and results.");

  //////////////////////////////////
  // Even with cmdp option "true", I get errors for having these
  //   arguments on the command line.  (On redsky build)
  // KDDKDD Should just be warnings, right?  Code should still work with these
  // KDDKDD params in the create-a-matrix file.  Better to have them where
  // KDDKDD they are used.
  int xdim=10;
  int ydim=10;
  int zdim=10;
  std::string matrixType("Laplace3D");

  cmdp.setOption("x", &xdim,
                "number of gridpoints in X dimension for "
                "mesh used to generate matrix.");
  cmdp.setOption("y", &ydim,
                "number of gridpoints in Y dimension for "
                "mesh used to generate matrix.");
  cmdp.setOption("z", &zdim,              
                "number of gridpoints in Z dimension for "
                "mesh used to generate matrix.");
  cmdp.setOption("matrix", &matrixType,
                "Matrix type: Laplace1D, Laplace2D, or Laplace3D");
  //////////////////////////////////

  cmdp.parse(narg, arg);

  RCP<UserInputForTests> uinput;

  if (inputFile != "")   // Input file specified; read a matrix
    uinput = rcp(new UserInputForTests(testDataFilePath, inputFile, comm, true));
  
  else                  // Let MueLu generate a default matrix
    uinput = rcp(new UserInputForTests(xdim, ydim, zdim, string(""), comm, true));

  RCP<SparseMatrix> origMatrix = uinput->getTpetraCrsMatrix();

  if (outputFile != "") {
    // Just a sanity check.
    Tpetra::MatrixMarket::Writer<SparseMatrix>::writeSparseFile(outputFile,
                                                origMatrix, verbose);
  }

  if (me == 0) 
    cout << "NumRows     = " << origMatrix->getGlobalNumRows() << endl
         << "NumNonzeros = " << origMatrix->getGlobalNumEntries() << endl
         << "NumProcs = " << comm->getSize() << endl;

  ////// Create a vector to use with the matrix.
  RCP<Vector> origVector, origProd;
  origProd   = Tpetra::createVector<Scalar,z2TestLO,z2TestGO>(
                                    origMatrix->getRangeMap());
  origVector = Tpetra::createVector<Scalar,z2TestLO,z2TestGO>(
                                    origMatrix->getDomainMap());
  origVector->randomize();

  ////// Specify problem parameters
  Teuchos::ParameterList params;
  Teuchos::ParameterList &partitioningParams = params.sublist("partitioning");
  
  partitioningParams.set("approach", "partition");
  partitioningParams.set("algorithm", "scotch");

  ////// Create an input adapter for the Tpetra matrix.
  SparseMatrixAdapter adapter(origMatrix);

  ////// Create and solve partitioning problem
  Zoltan2::PartitioningProblem<SparseMatrixAdapter> problem(&adapter, &params);

  try {
    if (me == 0) cout << "Calling solve() " << endl;
    problem.solve();
    if (me == 0) cout << "Done solve() " << endl;
  }
  catch (std::runtime_error &e) {
    cout << "Runtime exception returned from solve(): " << e.what();
    if (!strncmp(e.what(), "BUILD ERROR", 11)) {
      // Catching build errors as exceptions is OK in the tests
      cout << " PASS" << endl;
      return 0;
    }
    else {
      // All other runtime_errors are failures
      cout << " FAIL" << endl;
      return -1;
    }
  }
  catch (std::logic_error &e) {
    cout << "Logic exception returned from solve(): " << e.what()
         << " FAIL" << endl;
    return -1;
  }
  catch (std::bad_alloc &e) {
    cout << "Bad_alloc exception returned from solve(): " << e.what()
         << " FAIL" << endl;
    return -1;
  }
  catch (std::exception &e) {
    cout << "Unknown exception returned from solve(). " << e.what()
         << " FAIL" << endl;
    return -1;
  }

  ////// Basic metric checking of the partitioning solution
  ////// Not ordinarily done in application code; just doing it for testing here.
  size_t checkNparts = comm->getSize();
  
  size_t  checkLength = problem.getSolution().getLocalNumberOfIds();
  const zoltan2_partId_t *checkParts = problem.getSolution().getPartList();

  // Check for load balance
  size_t *countPerPart = new size_t[checkNparts];
  size_t *globalCountPerPart = new size_t[checkNparts];
  for (size_t i = 0; i < checkNparts; i++) countPerPart[i] = 0;
  for (size_t i = 0; i < checkLength; i++) {
    if (size_t(checkParts[i]) >= checkNparts) 
      cout << "Invalid Part:  FAIL" << endl;
    countPerPart[checkParts[i]]++;
  }
  Teuchos::reduceAll<int, size_t>(*comm, Teuchos::REDUCE_SUM, checkNparts,
                                  countPerPart, globalCountPerPart);

  size_t min = std::numeric_limits<std::size_t>::max();
  size_t max = 0;
  size_t sum = 0;
  size_t minrank = 0, maxrank = 0;
  for (size_t i = 0; i < checkNparts; i++) {
    if (globalCountPerPart[i] < min) {min = globalCountPerPart[i]; minrank = i;}
    if (globalCountPerPart[i] > max) {max = globalCountPerPart[i]; maxrank = i;}
    sum += globalCountPerPart[i];
  }
  delete [] countPerPart;
  delete [] globalCountPerPart;

  if (me == 0) {
    float avg = (float) sum / (float) checkNparts;
    cout << "Minimum load:  " << min << " on rank " << minrank << endl;
    cout << "Maximum load:  " << max << " on rank " << maxrank << endl;
    cout << "Average load:  " << avg << endl;
    cout << "Total load:    " << sum 
         << (sum != origMatrix->getGlobalNumRows()
                 ? "Work was lost; FAIL"
                 : " ")
         << endl;
    cout << "Imbalance:     " << max / avg << endl;
  }

  ////// Redistribute matrix and vector into new matrix and vector.
  if (me == 0) cout << "Redistributing matrix..." << endl;
  SparseMatrix *redistribMatrix;
  adapter.applyPartitioningSolution(*origMatrix, redistribMatrix,
                                    problem.getSolution());

  if (me == 0) cout << "Redistributing vectors..." << endl;
  Vector *redistribVector;
  std::vector<const scalar_t *> weights;
  std::vector<int> weightStrides;
  VectorAdapter adapterVector(origVector, weights, weightStrides);
  adapterVector.applyPartitioningSolution(*origVector, redistribVector,
                                          problem.getSolution());

  RCP<Vector> redistribProd;
  redistribProd = Tpetra::createVector<Scalar,z2TestLO,z2TestGO>(
                                       redistribMatrix->getRangeMap());


  ////// Verify that redistribution is "correct"; perform matvec with 
  ////// original and redistributed matrices/vectors and compare norms.

  if (me == 0) cout << "Matvec original..." << endl;
  origMatrix->apply(*origVector, *origProd);
  Scalar origNorm = origProd->norm2();
  if (me == 0)
    cout << "Norm of Original matvec prod:       " << origNorm << endl;

  if (me == 0) cout << "Matvec redistributed..." << endl;
  redistribMatrix->apply(*redistribVector, *redistribProd);
  Scalar redistribNorm = redistribProd->norm2();
  if (me == 0)
    cout << "Norm of Redistributed matvec prod:  " << redistribNorm << endl;

  if (redistribNorm > origNorm+epsilon || redistribNorm < origNorm-epsilon) 
    testReturn = 1;

  delete redistribVector;
  delete redistribMatrix;

  if (me == 0) {
    if (testReturn)
      std::cout << "Mat-Vec product changed; FAIL" << std::endl;
    else
      std::cout << "PASS" << std::endl;
  }

  return testReturn;
}