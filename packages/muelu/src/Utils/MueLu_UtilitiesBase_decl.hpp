// @HEADER
//
// ***********************************************************************
//
//        MueLu: A package for multigrid based preconditioning
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
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Andrey Prokopenko (aprokop@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#ifndef MUELU_UTILITIESBASE_DECL_HPP
#define MUELU_UTILITIESBASE_DECL_HPP

#ifndef _WIN32
#include <unistd.h> //necessary for "sleep" function in debugging methods (PauseForDebugging)
#endif

#include <string>

#include "MueLu_ConfigDefs.hpp"

#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Xpetra_BlockedCrsMatrix_fwd.hpp>
#include <Xpetra_CrsGraphFactory_fwd.hpp>
#include <Xpetra_CrsGraph_fwd.hpp>
#include <Xpetra_CrsMatrix_fwd.hpp>
#include <Xpetra_CrsMatrixWrap_fwd.hpp>
#include <Xpetra_Map_fwd.hpp>
#include <Xpetra_BlockedMap_fwd.hpp>
#include <Xpetra_MapFactory_fwd.hpp>
#include <Xpetra_Matrix_fwd.hpp>
#include <Xpetra_MatrixFactory_fwd.hpp>
#include <Xpetra_MultiVector_fwd.hpp>
#include <Xpetra_MultiVectorFactory_fwd.hpp>
#include <Xpetra_Operator_fwd.hpp>
#include <Xpetra_Vector_fwd.hpp>
#include <Xpetra_BlockedMultiVector.hpp>
#include <Xpetra_BlockedVector.hpp>
#include <Xpetra_VectorFactory_fwd.hpp>
#include <Xpetra_ExportFactory.hpp>

#include <Xpetra_Import.hpp>
#include <Xpetra_ImportFactory.hpp>
#include <Xpetra_MatrixMatrix.hpp>
#include <Xpetra_CrsGraph.hpp>
#include <Xpetra_CrsGraphFactory.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_StridedMap.hpp>

#include "MueLu_Exceptions.hpp"


namespace MueLu {

// MPI helpers
#define MueLu_sumAll(rcpComm, in, out)                                        \
    Teuchos::reduceAll(*rcpComm, Teuchos::REDUCE_SUM, in, Teuchos::outArg(out))
#define MueLu_minAll(rcpComm, in, out)                                        \
    Teuchos::reduceAll(*rcpComm, Teuchos::REDUCE_MIN, in, Teuchos::outArg(out))
#define MueLu_maxAll(rcpComm, in, out)                                        \
    Teuchos::reduceAll(*rcpComm, Teuchos::REDUCE_MAX, in, Teuchos::outArg(out))

  /*!
    @class Utilities
    @brief MueLu utility class.

    This class provides a number of static helper methods. Some are temporary and will eventually
    go away, while others should be moved to Xpetra.
  */
  template <class Scalar,
            class LocalOrdinal = DefaultLocalOrdinal,
            class GlobalOrdinal = DefaultGlobalOrdinal,
            class Node = DefaultNode>
  class UtilitiesBase {
  public:
#undef MUELU_UTILITIESBASE_SHORT
//#include "MueLu_UseShortNames.hpp"
  private:
    using CrsGraph = Xpetra::CrsGraph<LocalOrdinal,GlobalOrdinal,Node>;
    using CrsGraphFactory = Xpetra::CrsGraphFactory<LocalOrdinal,GlobalOrdinal,Node>;
    using CrsMatrixWrap = Xpetra::CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>;
    using CrsMatrix = Xpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>;
    using Matrix = Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>;
    using Vector = Xpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node>;
    using BlockedVector = Xpetra::BlockedVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>;
    using MultiVector = Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>;
    using BlockedMultiVector = Xpetra::BlockedMultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>;
    using BlockedMap = Xpetra::BlockedMap<LocalOrdinal,GlobalOrdinal,Node>;
    using Map = Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node>;
  public:
    typedef typename Teuchos::ScalarTraits<Scalar>::magnitudeType Magnitude;


    static RCP<Matrix>                Crs2Op(RCP<CrsMatrix> Op) {
      if (Op.is_null())
        return Teuchos::null;
      return rcp(new CrsMatrixWrap(Op));
    }

    /*! @brief Threshold a matrix

    Returns matrix filtered with a threshold value.

    NOTE -- it's assumed that A has been fillComplete'd.
    */
    static RCP<CrsMatrixWrap> GetThresholdedMatrix(const RCP<Matrix>& Ain, const Scalar threshold, const bool keepDiagonal=true, const GlobalOrdinal expectedNNZperRow=-1) {

      RCP<const Map> rowmap = Ain->getRowMap();
      RCP<const Map> colmap = Ain->getColMap();
      RCP<CrsMatrixWrap> Aout = rcp(new CrsMatrixWrap(rowmap, expectedNNZperRow <= 0 ? Ain->getGlobalMaxNumRowEntries() : expectedNNZperRow));
      // loop over local rows
      for(size_t row=0; row<Ain->getLocalNumRows(); row++)
      {
        size_t nnz = Ain->getNumEntriesInLocalRow(row);

        Teuchos::ArrayView<const LocalOrdinal> indices;
        Teuchos::ArrayView<const Scalar> vals;
        Ain->getLocalRowView(row, indices, vals);

        TEUCHOS_TEST_FOR_EXCEPTION(Teuchos::as<size_t>(indices.size()) != nnz, Exceptions::RuntimeError, "MueLu::ThresholdAFilterFactory::Build: number of nonzeros not equal to number of indices? Error.");

        Teuchos::ArrayRCP<GlobalOrdinal> indout(indices.size(),Teuchos::ScalarTraits<GlobalOrdinal>::zero());
        Teuchos::ArrayRCP<Scalar> valout(indices.size(),Teuchos::ScalarTraits<Scalar>::zero());
        size_t nNonzeros = 0;
        if (keepDiagonal) {
          GlobalOrdinal glbRow = rowmap->getGlobalElement(row);
          LocalOrdinal lclColIdx = colmap->getLocalElement(glbRow);
          for(size_t i=0; i<(size_t)indices.size(); i++) {
            if(Teuchos::ScalarTraits<Scalar>::magnitude(vals[i]) > Teuchos::ScalarTraits<Scalar>::magnitude(threshold) || indices[i]==lclColIdx) {
              indout[nNonzeros] = colmap->getGlobalElement(indices[i]); // LID -> GID (column)
              valout[nNonzeros] = vals[i];
              nNonzeros++;
            }
          }
        } else
          for(size_t i=0; i<(size_t)indices.size(); i++) {
            if(Teuchos::ScalarTraits<Scalar>::magnitude(vals[i]) > Teuchos::ScalarTraits<Scalar>::magnitude(threshold)) {
              indout[nNonzeros] = colmap->getGlobalElement(indices[i]); // LID -> GID (column)
              valout[nNonzeros] = vals[i];
              nNonzeros++;
            }
          }

        indout.resize(nNonzeros);
        valout.resize(nNonzeros);

        Aout->insertGlobalValues(Ain->getRowMap()->getGlobalElement(row), indout.view(0,indout.size()), valout.view(0,valout.size()));
      }
      Aout->fillComplete(Ain->getDomainMap(), Ain->getRangeMap());

      return Aout;
    }

    /*! @brief Threshold a graph

    Returns graph filtered with a threshold value.

    NOTE -- it's assumed that A has been fillComplete'd.
    */
    static RCP<Xpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node> > GetThresholdedGraph(const RCP<Matrix>& A, const Magnitude threshold, const GlobalOrdinal expectedNNZperRow=-1) {

      using STS = Teuchos::ScalarTraits<Scalar>;
      RCP<CrsGraph> sparsityPattern = CrsGraphFactory::Build(A->getRowMap(), expectedNNZperRow <= 0 ? A->getGlobalMaxNumRowEntries() : expectedNNZperRow);

      RCP<Vector> diag = GetMatrixOverlappedDiagonal(*A);
      ArrayRCP<const Scalar> D = diag->getData(0);

      for(size_t row=0; row<A->getLocalNumRows(); row++)
      {
        ArrayView<const LocalOrdinal> indices;
        ArrayView<const Scalar> vals;
        A->getLocalRowView(row, indices, vals);

        GlobalOrdinal globalRow = A->getRowMap()->getGlobalElement(row);
        LocalOrdinal col = A->getColMap()->getLocalElement(globalRow);

        const Scalar Dk = STS::magnitude(D[col]) > 0.0 ? STS::magnitude(D[col]) : 1.0;
        Array<GlobalOrdinal> indicesNew;

        for(size_t i=0; i<size_t(indices.size()); i++)
          // keep diagonal per default
          if(col == indices[i] || STS::magnitude(STS::squareroot(Dk)*vals[i]*STS::squareroot(Dk)) > STS::magnitude(threshold))
            indicesNew.append(A->getColMap()->getGlobalElement(indices[i]));

        sparsityPattern->insertGlobalIndices(globalRow, ArrayView<const GlobalOrdinal>(indicesNew.data(), indicesNew.length()));
      }
      sparsityPattern->fillComplete();

      return sparsityPattern;
    }

    /*! @brief Extract Matrix Diagonal

    Returns Matrix diagonal in ArrayRCP.

    NOTE -- it's assumed that A has been fillComplete'd.
    */
    static Teuchos::ArrayRCP<Scalar> GetMatrixDiagonal(const Matrix& A) {
      size_t numRows = A.getRowMap()->getLocalNumElements();
      Teuchos::ArrayRCP<Scalar> diag(numRows);
      Teuchos::ArrayView<const LocalOrdinal> cols;
      Teuchos::ArrayView<const Scalar> vals;
      for (size_t i = 0; i < numRows; ++i) {
        A.getLocalRowView(i, cols, vals);
        LocalOrdinal j = 0;
        for (; j < cols.size(); ++j) {
          if (Teuchos::as<size_t>(cols[j]) == i) {
            diag[i] = vals[j];
            break;
          }
        }
        if (j == cols.size()) {
          // Diagonal entry is absent
          diag[i] = Teuchos::ScalarTraits<Scalar>::zero();
        }
      }
      return diag;
    }

    /*! @brief Extract Matrix Diagonal

    Returns inverse of the Matrix diagonal in ArrayRCP.

    NOTE -- it's assumed that A has been fillComplete'd.
    */
    static RCP<Vector> GetMatrixDiagonalInverse(const Matrix& A, Magnitude tol = Teuchos::ScalarTraits<Scalar>::eps()*100, Scalar valReplacement = Teuchos::ScalarTraits<Scalar>::zero()) {
      Teuchos::TimeMonitor MM = *Teuchos::TimeMonitor::getNewTimer("UtilitiesBase::GetMatrixDiagonalInverse");

      RCP<const Map> rowMap = A.getRowMap();
      RCP<Vector> diag = Xpetra::VectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(rowMap,true);

      A.getLocalDiagCopy(*diag);

      RCP<Vector> inv = MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::GetInverse(diag, tol, valReplacement);

      return inv;
    }

    /*! @brief Extract Matrix Diagonal of lumped matrix

    Returns Matrix diagonal of lumped matrix in RCP<Vector>.

    NOTE -- it's assumed that A has been fillComplete'd.
    */
    static Teuchos::RCP<Vector> GetLumpedMatrixDiagonal(Matrix const & A, const bool doReciprocal = false,
                                                        Magnitude tol = Teuchos::ScalarTraits<Scalar>::magnitude(Teuchos::ScalarTraits<Scalar>::zero()),
                                                        Scalar valReplacement = Teuchos::ScalarTraits<Scalar>::zero(),
                                                        const bool replaceSingleEntryRowWithZero = false,
                                                        const bool useAverageAbsDiagVal = false) {

      typedef Teuchos::ScalarTraits<Scalar> TST;

      RCP<Vector> diag = Teuchos::null;
      const Scalar zero = TST::zero();
      const Scalar one = TST::one();
      const Scalar two = one + one;

      Teuchos::RCP<const Matrix> rcpA = Teuchos::rcpFromRef(A);

      RCP<const Xpetra::BlockedCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > bA =
          Teuchos::rcp_dynamic_cast<const Xpetra::BlockedCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(rcpA);
      if(bA == Teuchos::null) {
        RCP<const Map> rowMap = rcpA->getRowMap();
        diag = Xpetra::VectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(rowMap,true);
        ArrayRCP<Scalar> diagVals = diag->getDataNonConst(0);
        Teuchos::Array<Scalar> regSum(diag->getLocalLength());
        Teuchos::ArrayView<const LocalOrdinal> cols;
        Teuchos::ArrayView<const Scalar> vals;

        std::vector<int> nnzPerRow(rowMap->getLocalNumElements());

        //FIXME 2021-10-22 JHU   If this is called with doReciprocal=false, what should the correct behavior be?  Currently,
        //FIXME 2021-10-22 JHU   the diagonal entry is set to be the sum of the absolute values of the row entries.

        const Magnitude zeroMagn = TST::magnitude(zero);
        Magnitude avgAbsDiagVal = TST::magnitude(zero);
        int numDiagsEqualToOne = 0;
        for (size_t i = 0; i < rowMap->getLocalNumElements(); ++i) {
          nnzPerRow[i] = 0;
          rcpA->getLocalRowView(i, cols, vals);
          diagVals[i] = zero;
          for (LocalOrdinal j = 0; j < cols.size(); ++j) {
            regSum[i] += vals[j];
            const Magnitude rowEntryMagn = TST::magnitude(vals[j]);
            if (rowEntryMagn > zeroMagn)
              nnzPerRow[i]++;
            diagVals[i] += rowEntryMagn;
            if (static_cast<size_t>(cols[j]) == i)
              avgAbsDiagVal += rowEntryMagn;
          }
          if (nnzPerRow[i] == 1 && TST::magnitude(diagVals[i])==1.)
            numDiagsEqualToOne++;
        }
        if (useAverageAbsDiagVal)
          tol = TST::magnitude(100 * Teuchos::ScalarTraits<Scalar>::eps()) * (avgAbsDiagVal-numDiagsEqualToOne) / (rowMap->getLocalNumElements()-numDiagsEqualToOne);
        if (doReciprocal) {
          for (size_t i = 0; i < rowMap->getLocalNumElements(); ++i) {
            if (replaceSingleEntryRowWithZero && nnzPerRow[i] <= static_cast<int>(1))
              diagVals[i] = zero;
            else if ((diagVals[i] != zero) && (TST::magnitude(diagVals[i]) < TST::magnitude(two*regSum[i])))
              diagVals[i] = one / TST::magnitude((two*regSum[i]));
            else {
              if(TST::magnitude(diagVals[i]) > tol)
                diagVals[i] = one / diagVals[i];
              else {
                diagVals[i] = valReplacement;
              }
            }
          }
        }
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(doReciprocal, Xpetra::Exceptions::RuntimeError,
          "UtilitiesBase::GetLumpedMatrixDiagonal(): extracting reciprocal of diagonal of a blocked matrix is not supported");
        diag = Xpetra::VectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(bA->getRangeMapExtractor()->getFullMap(),true);

        for (size_t row = 0; row < bA->Rows(); ++row) {
          for (size_t col = 0; col < bA->Cols(); ++col) {
            if (!bA->getMatrix(row,col).is_null()) {
              // if we are in Thyra mode, but the block (row,row) is again a blocked operator, we have to use (pseudo) Xpetra-style GIDs with offset!
              bool bThyraMode = bA->getRangeMapExtractor()->getThyraMode() && (Teuchos::rcp_dynamic_cast<Xpetra::BlockedCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(bA->getMatrix(row,col)) == Teuchos::null);
              RCP<Vector> ddtemp = bA->getRangeMapExtractor()->ExtractVector(diag,row,bThyraMode);
              RCP<const Vector> dd = GetLumpedMatrixDiagonal(*(bA->getMatrix(row,col)));
              ddtemp->update(Teuchos::as<Scalar>(1.0),*dd,Teuchos::as<Scalar>(1.0));
              bA->getRangeMapExtractor()->InsertVector(ddtemp,row,diag,bThyraMode);
            }
          }
        }

      }

      return diag;
    }

    /*! @brief Return vector containing: max_{i\not=k}(-a_ik), for each for i in the matrix
     *
     * @param[in] A: input matrix
     * @ret: vector containing max_{i\not=k}(-a_ik)
    */

    static Teuchos::ArrayRCP<Magnitude> GetMatrixMaxMinusOffDiagonal(const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& A) {
      size_t numRows = A.getRowMap()->getLocalNumElements();
      Magnitude ZERO = Teuchos::ScalarTraits<Magnitude>::zero();
      Teuchos::ArrayRCP<Magnitude> maxvec(numRows);
      Teuchos::ArrayView<const LocalOrdinal> cols;
      Teuchos::ArrayView<const Scalar> vals;
      for (size_t i = 0; i < numRows; ++i) {
        A.getLocalRowView(i, cols, vals);
        Magnitude mymax = ZERO;
        for (LocalOrdinal j=0; j < cols.size(); ++j) {
          if (Teuchos::as<size_t>(cols[j]) != i) {
            mymax = std::max(mymax,-Teuchos::ScalarTraits<Scalar>::real(vals[j]));
          }
        }
        maxvec[i] = mymax;
      }
      return maxvec;
    }

    static Teuchos::ArrayRCP<Magnitude> GetMatrixMaxMinusOffDiagonal(const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& A, const Xpetra::Vector<LocalOrdinal,LocalOrdinal,GlobalOrdinal,Node> &BlockNumber) {
      TEUCHOS_TEST_FOR_EXCEPTION(!A.getColMap()->isSameAs(*BlockNumber.getMap()),std::runtime_error,"GetMatrixMaxMinusOffDiagonal: BlockNumber must match's A's column map.");
      
      Teuchos::ArrayRCP<const LocalOrdinal> block_id = BlockNumber.getData(0);

      size_t numRows = A.getRowMap()->getLocalNumElements();
      Magnitude ZERO = Teuchos::ScalarTraits<Magnitude>::zero();
      Teuchos::ArrayRCP<Magnitude> maxvec(numRows);
      Teuchos::ArrayView<const LocalOrdinal> cols;
      Teuchos::ArrayView<const Scalar> vals;
      for (size_t i = 0; i < numRows; ++i) {
        A.getLocalRowView(i, cols, vals);
        Magnitude mymax = ZERO;
        for (LocalOrdinal j=0; j < cols.size(); ++j) {
          if (Teuchos::as<size_t>(cols[j]) != i && block_id[i] == block_id[cols[j]]) {
            mymax = std::max(mymax,-Teuchos::ScalarTraits<Scalar>::real(vals[j]));
          }
        }
        //        printf("A(%d,:) row_scale(block) = %6.4e\n",(int)i,mymax);

        maxvec[i] = mymax;
      }
      return maxvec;
    }

    /*! @brief Return vector containing inverse of input vector
     *
     * @param[in] v: input vector
     * @param[in] tol: tolerance. If entries of input vector are smaller than tolerance they are replaced by valReplacement (see below). The default value for tol is 100*eps (machine precision)
     * @param[in] valReplacement: Value put in for undefined entries in output vector (default: 0.0)
     * @ret: vector containing inverse values of input vector v
    */
    static Teuchos::RCP<Vector> GetInverse(Teuchos::RCP<const Vector> v, Magnitude tol = Teuchos::ScalarTraits<Scalar>::eps()*100, Scalar valReplacement = Teuchos::ScalarTraits<Scalar>::zero()) {

      RCP<Vector> ret = Xpetra::VectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(v->getMap(),true);

      // check whether input vector "v" is a BlockedVector
      RCP<const BlockedVector> bv = Teuchos::rcp_dynamic_cast<const BlockedVector>(v);
      if(bv.is_null() == false) {
        RCP<BlockedVector> bret = Teuchos::rcp_dynamic_cast<BlockedVector>(ret);
        TEUCHOS_TEST_FOR_EXCEPTION(bret.is_null() == true, MueLu::Exceptions::RuntimeError,"MueLu::UtilitiesBase::GetInverse: return vector should be of type BlockedVector");
        RCP<const BlockedMap> bmap = bv->getBlockedMap();
        for(size_t r = 0; r < bmap->getNumMaps(); ++r) {
          RCP<const MultiVector> submvec = bv->getMultiVector(r,bmap->getThyraMode());
          RCP<const Vector> subvec = submvec->getVector(0);
          RCP<Vector> subvecinf = MueLu::UtilitiesBase<Scalar,LocalOrdinal,GlobalOrdinal,Node>::GetInverse(subvec,tol,valReplacement);
          bret->setMultiVector(r, subvecinf, bmap->getThyraMode());
        }
        return ret;
      }

      // v is an {Epetra,Tpetra}Vector: work with the underlying raw data
      ArrayRCP<Scalar> retVals = ret->getDataNonConst(0);
      ArrayRCP<const Scalar> inputVals = v->getData(0);
      for (size_t i = 0; i < v->getMap()->getLocalNumElements(); ++i) {
        if(Teuchos::ScalarTraits<Scalar>::magnitude(inputVals[i]) > tol)
          retVals[i] = Teuchos::ScalarTraits<Scalar>::one() / inputVals[i];
        else
          retVals[i] = valReplacement;
      }
      return ret;
    }

    /*! @brief Extract Overlapped Matrix Diagonal

    Returns overlapped Matrix diagonal in ArrayRCP.

    The local overlapped diagonal has an entry for each index in A's column map.
    NOTE -- it's assumed that A has been fillComplete'd.
    */
    static RCP<Vector> GetMatrixOverlappedDiagonal(const Matrix& A) {
      RCP<const Map> rowMap = A.getRowMap(), colMap = A.getColMap();

      // Undo block map (if we have one)
      RCP<const BlockedMap> browMap = Teuchos::rcp_dynamic_cast<const BlockedMap>(rowMap);
      if(!browMap.is_null()) rowMap = browMap->getMap();

      RCP<Vector> localDiag = Xpetra::VectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(rowMap);
      try {
         const CrsMatrixWrap* crsOp = dynamic_cast<const CrsMatrixWrap*>(&A);
         if (crsOp == NULL) {
           throw Exceptions::RuntimeError("cast to CrsMatrixWrap failed");
         }
         Teuchos::ArrayRCP<size_t> offsets;
         crsOp->getLocalDiagOffsets(offsets);
         crsOp->getLocalDiagCopy(*localDiag,offsets());
      }
      catch (...) {
        ArrayRCP<Scalar>   localDiagVals = localDiag->getDataNonConst(0);
        Teuchos::ArrayRCP<Scalar> diagVals = GetMatrixDiagonal(A);
        for (LocalOrdinal i = 0; i < localDiagVals.size(); i++)
          localDiagVals[i] = diagVals[i];
        localDiagVals = diagVals = null;
      }

      RCP<Vector> diagonal = Xpetra::VectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(colMap);
      RCP< const Xpetra::Import<LocalOrdinal,GlobalOrdinal,Node> > importer;
      importer = A.getCrsGraph()->getImporter();
      if (importer == Teuchos::null) {
        importer = Xpetra::ImportFactory<LocalOrdinal,GlobalOrdinal,Node>::Build(rowMap, colMap);
      }
      diagonal->doImport(*localDiag, *(importer), Xpetra::INSERT);
      return diagonal;
    }


    /*! @brief Extract Overlapped Matrix Deleted Rowsum

    Returns overlapped Matrix deleted Rowsum in ArrayRCP.

    The local overlapped deleted Rowsum has an entry for each index in A's column map.
    NOTE -- it's assumed that A has been fillComplete'd.
    */
    static RCP<Vector> GetMatrixOverlappedDeletedRowsum(const Matrix& A) {
      using LO  = LocalOrdinal;
      using GO  = GlobalOrdinal;
      using SC  = Scalar;
      using STS = typename Teuchos::ScalarTraits<SC>;

      // Undo block map (if we have one)
      RCP<const Map> rowMap = A.getRowMap(), colMap = A.getColMap();
      RCP<const BlockedMap> browMap = Teuchos::rcp_dynamic_cast<const BlockedMap>(rowMap);
      if(!browMap.is_null()) rowMap = browMap->getMap();

      RCP<Vector> local   = Xpetra::VectorFactory<SC,LO,GO,Node>::Build(rowMap);
      RCP<Vector> ghosted = Xpetra::VectorFactory<SC,LO,GO,Node>::Build(colMap,true);
      ArrayRCP<SC> localVals = local->getDataNonConst(0);

      for (LO row = 0; row < static_cast<LO>(A.getRowMap()->getLocalNumElements()); ++row) {
	size_t nnz = A.getNumEntriesInLocalRow(row);
	ArrayView<const LO> indices;
	ArrayView<const SC> vals;
	A.getLocalRowView(row, indices, vals);

	SC si = STS::zero();

	for (LO colID = 0; colID < static_cast<LO>(nnz); colID++) {
	  if(indices[colID] != row) {
	    si += vals[colID];
	  }
	}
	localVals[row] = si;
      }

      RCP< const Xpetra::Import<LO,GO,Node> > importer;
      importer = A.getCrsGraph()->getImporter();
      if (importer == Teuchos::null) {
        importer = Xpetra::ImportFactory<LO,GO,Node>::Build(rowMap, colMap);
      }
      ghosted->doImport(*local, *(importer), Xpetra::INSERT);
      return ghosted;
    }



    static RCP<Xpetra::Vector<Magnitude,LocalOrdinal,GlobalOrdinal,Node> >
    GetMatrixOverlappedAbsDeletedRowsum(const Matrix& A) {
      RCP<const Map> rowMap = A.getRowMap(), colMap = A.getColMap();
      using STS = typename Teuchos::ScalarTraits<Scalar>;
      using MTS = typename Teuchos::ScalarTraits<Magnitude>;
      using MT  = Magnitude;
      using LO  = LocalOrdinal;
      using GO  = GlobalOrdinal;
      using SC  = Scalar;
      using RealValuedVector = Xpetra::Vector<MT,LO,GO,Node>;

      // Undo block map (if we have one)
      RCP<const BlockedMap> browMap = Teuchos::rcp_dynamic_cast<const BlockedMap>(rowMap);
      if(!browMap.is_null()) rowMap = browMap->getMap();

      RCP<RealValuedVector> local     = Xpetra::VectorFactory<MT,LO,GO,Node>::Build(rowMap);
      RCP<RealValuedVector> ghosted   = Xpetra::VectorFactory<MT,LO,GO,Node>::Build(colMap,true);
      ArrayRCP<MT>          localVals = local->getDataNonConst(0);

      for (LO rowIdx = 0; rowIdx < static_cast<LO>(A.getRowMap()->getLocalNumElements()); ++rowIdx) {
        size_t nnz = A.getNumEntriesInLocalRow(rowIdx);
        ArrayView<const LO> indices;
        ArrayView<const SC> vals;
        A.getLocalRowView(rowIdx, indices, vals);

        MT si = MTS::zero();

        for (LO colID = 0; colID < static_cast<LO>(nnz); ++colID) {
          if(indices[colID] != rowIdx) {
            si += STS::magnitude(vals[colID]);
          }
        }
        localVals[rowIdx] = si;
      }

      RCP< const Xpetra::Import<LO,GO,Node> > importer;
      importer = A.getCrsGraph()->getImporter();
      if (importer == Teuchos::null) {
        importer = Xpetra::ImportFactory<LO,GO,Node>::Build(rowMap, colMap);
      }
      ghosted->doImport(*local, *(importer), Xpetra::INSERT);
      return ghosted;
    }



    // TODO: should NOT return an Array. Definition must be changed to:
    // - ArrayRCP<> ResidualNorm(Matrix const &Op, MultiVector const &X, MultiVector const &RHS)
    // or
    // - void ResidualNorm(Matrix const &Op, MultiVector const &X, MultiVector const &RHS, Array &)
    static Teuchos::Array<Magnitude> ResidualNorm(const Xpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Op, const MultiVector& X, const MultiVector& RHS) {
      TEUCHOS_TEST_FOR_EXCEPTION(X.getNumVectors() != RHS.getNumVectors(), Exceptions::RuntimeError, "Number of solution vectors != number of right-hand sides")
       const size_t numVecs = X.getNumVectors();
       RCP<MultiVector> RES = Residual(Op, X, RHS);
       Teuchos::Array<Magnitude> norms(numVecs);
       RES->norm2(norms);
       return norms;
    }

    static Teuchos::Array<Magnitude> ResidualNorm(const Xpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Op, const MultiVector& X, const MultiVector& RHS, MultiVector & Resid) {
      TEUCHOS_TEST_FOR_EXCEPTION(X.getNumVectors() != RHS.getNumVectors(), Exceptions::RuntimeError, "Number of solution vectors != number of right-hand sides")
       const size_t numVecs = X.getNumVectors();
       Residual(Op,X,RHS,Resid);
       Teuchos::Array<Magnitude> norms(numVecs);
       Resid.norm2(norms);
       return norms;
    }

    static RCP<MultiVector> Residual(const Xpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Op, const MultiVector& X, const MultiVector& RHS) {
      TEUCHOS_TEST_FOR_EXCEPTION(X.getNumVectors() != RHS.getNumVectors(), Exceptions::RuntimeError, "Number of solution vectors != number of right-hand sides")
        const size_t numVecs = X.getNumVectors();
        // TODO Op.getRangeMap should return a BlockedMap if it is a BlockedCrsOperator
        RCP<MultiVector> RES = Xpetra::MultiVectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(RHS.getMap(), numVecs, false); // no need to initialize to zero
        Op.residual(X,RHS,*RES);
        return RES;
    }


    static void Residual(const Xpetra::Operator<Scalar,LocalOrdinal,GlobalOrdinal,Node>& Op, const MultiVector& X, const MultiVector& RHS, MultiVector & Resid) {
      TEUCHOS_TEST_FOR_EXCEPTION(X.getNumVectors() != RHS.getNumVectors(), Exceptions::RuntimeError, "Number of solution vectors != number of right-hand sides");
      TEUCHOS_TEST_FOR_EXCEPTION(Resid.getNumVectors() != RHS.getNumVectors(), Exceptions::RuntimeError, "Number of residual vectors != number of right-hand sides");
      Op.residual(X,RHS,Resid);
    }


    /*! @brief Power method.

    @param A matrix
    @param scaleByDiag if true, estimate the largest eigenvalue of \f$ D^; A \f$.
    @param niters maximum number of iterations
    @param tolerance stopping tolerance
    @verbose if true, print iteration information
    @seed  seed for randomizing initial guess

    (Shamelessly grabbed from tpetra/examples.)
    */
    static Scalar PowerMethod(const Matrix& A, bool scaleByDiag = true,
                              LocalOrdinal niters = 10, Magnitude tolerance = 1e-2, bool verbose = false, unsigned int seed = 123) {
      TEUCHOS_TEST_FOR_EXCEPTION(!(A.getRangeMap()->isSameAs(*(A.getDomainMap()))), Exceptions::Incompatible,
          "Utils::PowerMethod: operator must have domain and range maps that are equivalent.");

      // power iteration
      RCP<Vector> diagInvVec;
      if (scaleByDiag) {
        RCP<Vector> diagVec = Xpetra::VectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(A.getRowMap());
        A.getLocalDiagCopy(*diagVec);
        diagInvVec = Xpetra::VectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(A.getRowMap());
        diagInvVec->reciprocal(*diagVec);
      }

      Scalar lambda = PowerMethod(A, diagInvVec, niters, tolerance, verbose, seed);
      return lambda;
    }

    /*! @brief Power method.

    @param A matrix
    @param diagInvVec reciprocal of matrix diagonal
    @param niters maximum number of iterations
    @param tolerance stopping tolerance
    @verbose if true, print iteration information
    @seed  seed for randomizing initial guess

    (Shamelessly grabbed from tpetra/examples.)
    */
    static Scalar PowerMethod(const Matrix& A, const RCP<Vector> &diagInvVec,
                              LocalOrdinal niters = 10, Magnitude tolerance = 1e-2, bool verbose = false, unsigned int seed = 123) {
      TEUCHOS_TEST_FOR_EXCEPTION(!(A.getRangeMap()->isSameAs(*(A.getDomainMap()))), Exceptions::Incompatible,
          "Utils::PowerMethod: operator must have domain and range maps that are equivalent.");

      // Create three vectors, fill z with random numbers
      RCP<Vector> q = Xpetra::VectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(A.getDomainMap());
      RCP<Vector> r = Xpetra::VectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(A.getRangeMap());
      RCP<Vector> z = Xpetra::VectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(A.getRangeMap());

      z->setSeed(seed);  // seed random number generator
      z->randomize(true);// use Xpetra implementation: -> same results for Epetra and Tpetra

      Teuchos::Array<Magnitude> norms(1);

      typedef Teuchos::ScalarTraits<Scalar> STS;

      const Scalar zero = STS::zero(), one = STS::one();

      Scalar lambda = zero;
      Magnitude residual = STS::magnitude(zero);

      // power iteration
      for (int iter = 0; iter < niters; ++iter) {
        z->norm2(norms);                                  // Compute 2-norm of z
        q->update(one/norms[0], *z, zero);                // Set q = z / normz
        A.apply(*q, *z);                                  // Compute z = A*q
        if (diagInvVec != Teuchos::null)
          z->elementWiseMultiply(one, *diagInvVec, *z, zero);
        lambda = q->dot(*z);                              // Approximate maximum eigenvalue: lamba = dot(q,z)

        if (iter % 100 == 0 || iter + 1 == niters) {
          r->update(1.0, *z, -lambda, *q, zero);          // Compute A*q - lambda*q
          r->norm2(norms);
          residual = STS::magnitude(norms[0] / lambda);
          if (verbose) {
            std::cout << "Iter = " << iter
                      << "  Lambda = " << lambda
                      << "  Residual of A*q - lambda*q = " << residual
                      << std::endl;
          }
        }
        if (residual < tolerance)
          break;
      }
      return lambda;
    }

    static RCP<Teuchos::FancyOStream> MakeFancy(std::ostream& os) {
      RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(os));
      return fancy;
    }

    /*! @brief Squared distance between two rows in a multivector

       Used for coordinate vectors.
    */
    static typename Teuchos::ScalarTraits<Scalar>::magnitudeType Distance2(const Teuchos::Array<Teuchos::ArrayRCP<const Scalar>> &v, LocalOrdinal i0, LocalOrdinal i1) {
      const size_t numVectors = v.size();

      Scalar d = Teuchos::ScalarTraits<Scalar>::zero();
      for (size_t j = 0; j < numVectors; j++) {
        d += (v[j][i0] - v[j][i1])*(v[j][i0] - v[j][i1]);
      }
      return Teuchos::ScalarTraits<Scalar>::magnitude(d);
    }

    /*! @brief Weighted squared distance between two rows in a multivector

       Used for coordinate vectors.
    */
    static typename Teuchos::ScalarTraits<Scalar>::magnitudeType Distance2(const Teuchos::ArrayView<double> & weight,const Teuchos::Array<Teuchos::ArrayRCP<const Scalar>> &v, LocalOrdinal i0, LocalOrdinal i1) {
      const size_t numVectors = v.size();
      using MT = typename Teuchos::ScalarTraits<Scalar>::magnitudeType;

      Scalar d = Teuchos::ScalarTraits<Scalar>::zero();
      for (size_t j = 0; j < numVectors; j++) {
        d += Teuchos::as<MT>(weight[j])*(v[j][i0] - v[j][i1])*(v[j][i0] - v[j][i1]);
      }
      return Teuchos::ScalarTraits<Scalar>::magnitude(d);
    }


    /*! @brief Detect Dirichlet rows

        The routine assumes, that if there is only one nonzero per row, it is on the diagonal and therefore a DBC.
        This is safe for most of our applications, but one should be aware of that.

        There is an alternative routine (see DetectDirichletRowsExt)

        @param[in] A matrix
        @param[in] tol If a row entry's magnitude is less than or equal to this tolerance, the entry is treated as zero.

        @return boolean array.  The ith entry is true iff row i is a Dirichlet row.
    */
    static Teuchos::ArrayRCP<const bool> DetectDirichletRows(const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& A, const Magnitude& tol = Teuchos::ScalarTraits<Scalar>::zero(), bool count_twos_as_dirichlet=false) {
      LocalOrdinal numRows = A.getLocalNumRows();
      typedef Teuchos::ScalarTraits<Scalar> STS;
      ArrayRCP<bool> boundaryNodes(numRows, true);
      if (count_twos_as_dirichlet) {
        for (LocalOrdinal row = 0; row < numRows; row++) {
          ArrayView<const LocalOrdinal> indices;
          ArrayView<const Scalar> vals;
          A.getLocalRowView(row, indices, vals);
          size_t nnz = A.getNumEntriesInLocalRow(row);
          if (nnz > 2) {
            size_t col;
            for (col = 0; col < nnz; col++)
              if ( (indices[col] != row) && STS::magnitude(vals[col]) > tol) {
                if (!boundaryNodes[row])
                  break;
                boundaryNodes[row] = false;
              }
            if (col == nnz)
              boundaryNodes[row] = true;
          }
        }
      } else {
        for (LocalOrdinal row = 0; row < numRows; row++) {
          ArrayView<const LocalOrdinal> indices;
          ArrayView<const Scalar> vals;
          A.getLocalRowView(row, indices, vals);
          size_t nnz = A.getNumEntriesInLocalRow(row);
          if (nnz > 1)
            for (size_t col = 0; col < nnz; col++)
              if ( (indices[col] != row) && STS::magnitude(vals[col]) > tol) {
                boundaryNodes[row] = false;
                break;
              }
        }
      }
      return boundaryNodes;
    }


    /*! @brief Detect Dirichlet rows (extended version)

        Look at each matrix row and mark it as Dirichlet if there is only one
        "not small" nonzero on the diagonal. In determining whether a nonzero
        is "not small" use
               \f abs(A(i,j)) / sqrt(abs(diag[i]*diag[j])) > tol

        @param[in] A matrix
        @param[in/out] bHasZeroDiagonal Reference to boolean variable. Returns true if there is a zero on the diagonal in the local part of the Matrix. Otherwise it is false. Different processors might return a different value. There is no global reduction!
        @param[in] tol If a row entry's magnitude is less than or equal to this tolerance, the entry is treated as zero.
        @return boolean array.  The ith entry is true iff row i is a Dirichlet row.
    */
    static Teuchos::ArrayRCP<const bool> DetectDirichletRowsExt(const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& A, bool & bHasZeroDiagonal, const Magnitude& tol = Teuchos::ScalarTraits<Scalar>::zero()) {

      // assume that there is no zero diagonal in matrix
      bHasZeroDiagonal = false;

      Teuchos::RCP<Vector> diagVec = Xpetra::VectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(A.getRowMap());
      A.getLocalDiagCopy(*diagVec);
      Teuchos::ArrayRCP< const Scalar > diagVecData = diagVec->getData(0);

      LocalOrdinal numRows = A.getLocalNumRows();
      typedef Teuchos::ScalarTraits<Scalar> STS;
      ArrayRCP<bool> boundaryNodes(numRows, false);
      for (LocalOrdinal row = 0; row < numRows; row++) {
        ArrayView<const LocalOrdinal> indices;
        ArrayView<const Scalar> vals;
        A.getLocalRowView(row, indices, vals);
        size_t nnz = 0; // collect nonzeros in row (excluding the diagonal)
        bool bHasDiag = false;
        for (decltype(indices.size()) col = 0; col < indices.size(); col++) {
          if ( indices[col] != row) {
            if (STS::magnitude(vals[col] / STS::magnitude(sqrt(STS::magnitude(diagVecData[row]) * STS::magnitude(diagVecData[col])))   ) > tol) {
              nnz++;
            }
          } else bHasDiag = true; // found a diagonal entry
        }
        if (bHasDiag == false) bHasZeroDiagonal = true; // we found at least one row without a diagonal
        else if(nnz == 0) boundaryNodes[row] = true;
      }
      return boundaryNodes;
    }

    /*! @brief Find non-zero values in an ArrayRCP
      Compares the value to 2 * machine epsilon

      @param[in]  vals - ArrayRCP<const Scalar> of values to be tested
      @param[out] nonzeros - ArrayRCP<bool> of true/false values for whether each entry in vals is nonzero
    */
    
    static void FindNonZeros(const Teuchos::ArrayRCP<const Scalar> vals,
                             Teuchos::ArrayRCP<bool> nonzeros) {
      TEUCHOS_ASSERT(vals.size() == nonzeros.size());
      typedef typename Teuchos::ScalarTraits<Scalar>::magnitudeType magnitudeType;
      const magnitudeType eps = 2.0*Teuchos::ScalarTraits<magnitudeType>::eps();
      for(size_t i=0; i<static_cast<size_t>(vals.size()); i++) {
        nonzeros[i] = (Teuchos::ScalarTraits<Scalar>::magnitude(vals[i]) > eps);
      }
    }

    /*! @brief Detects Dirichlet columns & domains from a list of Dirichlet rows

      @param[in] A - Matrix on which to apply Dirichlet column detection
      @param[in] dirichletRows - ArrayRCP<bool> of indicators as to which rows are Dirichlet
      @param[out] dirichletCols - ArrayRCP<bool> of indicators as to which cols are Dirichlet
      @param[out] dirichletDomain - ArrayRCP<bool> of indicators as to which domains are Dirichlet
    */

    static void DetectDirichletColsAndDomains(const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& A,
                                              const Teuchos::ArrayRCP<bool>& dirichletRows,
                                              Teuchos::ArrayRCP<bool> dirichletCols,
                                              Teuchos::ArrayRCP<bool> dirichletDomain) {
      const Scalar one = Teuchos::ScalarTraits<Scalar>::one();
      RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > domMap = A .getDomainMap();
      RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > rowMap = A.getRowMap();
      RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > colMap = A.getColMap();
      TEUCHOS_ASSERT(static_cast<size_t>(dirichletRows.size()) == rowMap->getLocalNumElements());
      TEUCHOS_ASSERT(static_cast<size_t>(dirichletCols.size()) == colMap->getLocalNumElements());
      TEUCHOS_ASSERT(static_cast<size_t>(dirichletDomain.size()) == domMap->getLocalNumElements());
      RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > myColsToZero = Xpetra::MultiVectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(colMap, 1, /*zeroOut=*/true);
      // Find all local column indices that are in Dirichlet rows, record in myColsToZero as 1.0
      for(size_t i=0; i<(size_t) dirichletRows.size(); i++) {
        if (dirichletRows[i]) {
          ArrayView<const LocalOrdinal> indices;
          ArrayView<const Scalar> values;
          A.getLocalRowView(i,indices,values);
          for(size_t j=0; j<static_cast<size_t>(indices.size()); j++)
            myColsToZero->replaceLocalValue(indices[j],0,one);
        }
      }
      
      RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > globalColsToZero;
      RCP<const Xpetra::Import<LocalOrdinal,GlobalOrdinal,Node> > importer = A.getCrsGraph()->getImporter();
      if (!importer.is_null()) {
        globalColsToZero = Xpetra::MultiVectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(domMap, 1, /*zeroOut=*/true);
        // export to domain map
        globalColsToZero->doExport(*myColsToZero,*importer,Xpetra::ADD);
        // import to column map
      myColsToZero->doImport(*globalColsToZero,*importer,Xpetra::INSERT);
      }
      else
        globalColsToZero = myColsToZero;
      
      FindNonZeros(globalColsToZero->getData(0),dirichletDomain);
      FindNonZeros(myColsToZero->getData(0),dirichletCols);
    }



   /*! @brief Apply Rowsum Criterion

        Flags a row i as dirichlet if:
    
        \sum_{j\not=i} A_ij > A_ii * tol

        @param[in] A matrix
        @param[in] rowSumTol See above
        @param[in/out] dirichletRows boolean array.  The ith entry is true if the above criterion is satisfied (or if it was already set to true)

    */
    static void                                                                  ApplyRowSumCriterion(const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& A, const Magnitude rowSumTol, Teuchos::ArrayRCP<bool>& dirichletRows) {
      typedef Teuchos::ScalarTraits<Scalar> STS;
      typedef typename Teuchos::ScalarTraits<Scalar>::magnitudeType MT;
      typedef Teuchos::ScalarTraits<MT> MTS;
      RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node>> rowmap = A.getRowMap();
      for (LocalOrdinal row = 0; row < Teuchos::as<LocalOrdinal>(rowmap->getLocalNumElements()); ++row) {
        size_t nnz = A.getNumEntriesInLocalRow(row);
        ArrayView<const LocalOrdinal> indices;
        ArrayView<const Scalar> vals;
        A.getLocalRowView(row, indices, vals);
        
        Scalar rowsum = STS::zero();
        Scalar diagval = STS::zero();

        for (LocalOrdinal colID = 0; colID < Teuchos::as<LocalOrdinal>(nnz); colID++) {
          LocalOrdinal col = indices[colID];
          if (row == col)
            diagval = vals[colID];
          rowsum += vals[colID];
        }
        //        printf("A(%d,:) row_sum(point) = %6.4e\n",row,rowsum);
        if (rowSumTol < MTS::one() && STS::magnitude(rowsum) > STS::magnitude(diagval) * rowSumTol) {
          //printf("Row %d triggers rowsum\n",(int)row);
          dirichletRows[row] = true;
        }
      }
    }

    static void ApplyRowSumCriterion(const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& A, const Xpetra::Vector<LocalOrdinal,LocalOrdinal,GlobalOrdinal,Node> &BlockNumber, const Magnitude rowSumTol, Teuchos::ArrayRCP<bool>& dirichletRows) {
      typedef Teuchos::ScalarTraits<Scalar> STS;
      typedef typename Teuchos::ScalarTraits<Scalar>::magnitudeType MT;
      typedef Teuchos::ScalarTraits<MT> MTS;
      RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > rowmap = A.getRowMap();

      TEUCHOS_TEST_FOR_EXCEPTION(!A.getColMap()->isSameAs(*BlockNumber.getMap()),std::runtime_error,"ApplyRowSumCriterion: BlockNumber must match's A's column map.");
      
      Teuchos::ArrayRCP<const LocalOrdinal> block_id = BlockNumber.getData(0);
      for (LocalOrdinal row = 0; row < Teuchos::as<LocalOrdinal>(rowmap->getLocalNumElements()); ++row) {
        size_t nnz = A.getNumEntriesInLocalRow(row);
        ArrayView<const LocalOrdinal> indices;
        ArrayView<const Scalar> vals;
        A.getLocalRowView(row, indices, vals);
        
        Scalar rowsum = STS::zero();
        Scalar diagval = STS::zero();
        for (LocalOrdinal colID = 0; colID < Teuchos::as<LocalOrdinal>(nnz); colID++) {
          LocalOrdinal col = indices[colID];
          if (row == col)
            diagval = vals[colID];
          if(block_id[row] == block_id[col])
            rowsum += vals[colID];
        }

        //        printf("A(%d,:) row_sum(block) = %6.4e\n",row,rowsum);
        if (rowSumTol < MTS::one() && STS::magnitude(rowsum) > STS::magnitude(diagval) * rowSumTol) {
          //printf("Row %d triggers rowsum\n",(int)row);
          dirichletRows[row] = true;
        }
      }
    }



    /*! @brief Detect Dirichlet columns based on Dirichlet rows

        The routine finds all column indices that are in Dirichlet rows, where Dirichlet rows are described by dirichletRows,
        as returned by DetectDirichletRows.

        @param[in] A matrix
        @param[in] dirichletRows array of Dirichlet rows.

        @return boolean array.  The ith entry is true iff row i is a Dirichlet column.
    */
    static Teuchos::ArrayRCP<const bool> DetectDirichletCols(const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& A,
                                                             const Teuchos::ArrayRCP<const bool>& dirichletRows) {
      Scalar zero = Teuchos::ScalarTraits<Scalar>::zero();
      Scalar one = Teuchos::ScalarTraits<Scalar>::one();
      Teuchos::RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > domMap = A.getDomainMap();
      Teuchos::RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > colMap = A.getColMap();
      Teuchos::RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > myColsToZero = Xpetra::MultiVectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(colMap,1);
      myColsToZero->putScalar(zero);
      // Find all local column indices that are in Dirichlet rows, record in myColsToZero as 1.0
      for(size_t i=0; i<(size_t) dirichletRows.size(); i++) {
        if (dirichletRows[i]) {
          Teuchos::ArrayView<const LocalOrdinal> indices;
          Teuchos::ArrayView<const Scalar> values;
          A.getLocalRowView(i,indices,values);
          for(size_t j=0; j<static_cast<size_t>(indices.size()); j++)
            myColsToZero->replaceLocalValue(indices[j],0,one);
        }
      }

      Teuchos::RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > globalColsToZero = Xpetra::MultiVectorFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(domMap,1);
      globalColsToZero->putScalar(zero);
      Teuchos::RCP<Xpetra::Export<LocalOrdinal,GlobalOrdinal,Node> > exporter = Xpetra::ExportFactory<LocalOrdinal,GlobalOrdinal,Node>::Build(colMap,domMap);
      // export to domain map
      globalColsToZero->doExport(*myColsToZero,*exporter,Xpetra::ADD);
      // import to column map
      myColsToZero->doImport(*globalColsToZero,*exporter,Xpetra::INSERT);
      Teuchos::ArrayRCP<const Scalar> myCols = myColsToZero->getData(0);
      Teuchos::ArrayRCP<bool> dirichletCols(colMap->getLocalNumElements(), true);
      Magnitude eps = Teuchos::ScalarTraits<Magnitude>::eps();
      for(size_t i=0; i<colMap->getLocalNumElements(); i++) {
        dirichletCols[i] = Teuchos::ScalarTraits<Scalar>::magnitude(myCols[i])>2.0*eps;
      }
      return dirichletCols;
    }


    /*! @brief Frobenius inner product of two matrices

       Used in energy minimization algorithms
    */
    static Scalar Frobenius(const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& A, const Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>& B) {
      // We check only row maps. Column may be different. One would hope that they are the same, as we typically
      // calculate frobenius norm of the specified sparsity pattern with an updated matrix from the previous step,
      // but matrix addition, even when one is submatrix of the other, changes column map (though change may be as
      // simple as couple of elements swapped)
      TEUCHOS_TEST_FOR_EXCEPTION(!A.getRowMap()->isSameAs(*B.getRowMap()),   Exceptions::Incompatible, "MueLu::CGSolver::Frobenius: row maps are incompatible");
      TEUCHOS_TEST_FOR_EXCEPTION(!A.isFillComplete() || !B.isFillComplete(), Exceptions::RuntimeError, "Matrices must be fill completed");

      const Map& AColMap = *A.getColMap();
      const Map& BColMap = *B.getColMap();

      Teuchos::ArrayView<const LocalOrdinal> indA, indB;
      Teuchos::ArrayView<const Scalar>       valA, valB;
      size_t nnzA = 0, nnzB = 0;

      // We use a simple algorithm
      // for each row we fill valBAll array with the values in the corresponding row of B
      // as such, it serves as both sorted array and as storage, so we don't need to do a
      // tricky problem: "find a value in the row of B corresponding to the specific GID"
      // Once we do that, we translate LID of entries of row of A to LID of B, and multiply
      // corresponding entries.
      // The algorithm should be reasonably cheap, as it does not sort anything, provided
      // that getLocalElement and getGlobalElement functions are reasonably effective. It
      // *is* possible that the costs are hidden in those functions, but if maps are close
      // to linear maps, we should be fine
      Teuchos::Array<Scalar> valBAll(BColMap.getLocalNumElements());

      LocalOrdinal  invalid = Teuchos::OrdinalTraits<LocalOrdinal>::invalid();
      Scalar        zero    = Teuchos::ScalarTraits<Scalar>       ::zero(),    f = zero, gf;
      size_t numRows = A.getLocalNumRows();
      for (size_t i = 0; i < numRows; i++) {
        A.getLocalRowView(i, indA, valA);
        B.getLocalRowView(i, indB, valB);
        nnzA = indA.size();
        nnzB = indB.size();

        // Set up array values
        for (size_t j = 0; j < nnzB; j++)
          valBAll[indB[j]] = valB[j];

        for (size_t j = 0; j < nnzA; j++) {
          // The cost of the whole Frobenius dot product function depends on the
          // cost of the getLocalElement and getGlobalElement functions here.
          LocalOrdinal ind = BColMap.getLocalElement(AColMap.getGlobalElement(indA[j]));
          if (ind != invalid)
            f += valBAll[ind] * valA[j];
        }

        // Clean up array values
        for (size_t j = 0; j < nnzB; j++)
          valBAll[indB[j]] = zero;
      }

      MueLu_sumAll(AColMap.getComm(), f, gf);

      return gf;
    }

    /*! @brief Set seed for random number generator.

      Distribute the seeds evenly in [1,INT_MAX-1].  This guarantees nothing
      about where random number streams on difference processes will intersect.
      This does avoid overflow situations in parallel when multiplying by a PID.
      It also avoids the pathological case of having the *same* random number stream
      on each process.
    */

    static void SetRandomSeed(const Teuchos::Comm<int> &comm) {
      // Distribute the seeds evenly in [1,maxint-1].  This guarantees nothing
      // about where in random number stream we are, but avoids overflow situations
      // in parallel when multiplying by a PID.  It would be better to use
      // a good parallel random number generator.
      double one = 1.0;
      int maxint = INT_MAX; //= 2^31-1 = 2147483647 for 32-bit integers
      int mySeed = Teuchos::as<int>((maxint-1) * (one -(comm.getRank()+1)/(comm.getSize()+one)) );
      if (mySeed < 1 || mySeed == maxint) {
        std::ostringstream errStr;
        errStr << "Error detected with random seed = " << mySeed << ". It should be in the interval [1,2^31-2].";
        throw Exceptions::RuntimeError(errStr.str());
      }
      std::srand(mySeed);
      // For Tpetra, we could use Kokkos' random number generator here.
      Teuchos::ScalarTraits<Scalar>::seedrandom(mySeed);
      // Epetra
      //   MultiVector::Random() -> Epetra_Util::RandomDouble() -> Epetra_Utils::RandomInt()
      // Its own random number generator, based on Seed_. Seed_ is initialized in Epetra_Util constructor with std::rand()
      // So our setting std::srand() affects that too
    }



    // Finds the OAZ Dirichlet rows for this matrix
    // so far only used in IntrepidPCoarsenFactory
    // TODO check whether we can use DetectDirichletRows instead
    static void FindDirichletRows(Teuchos::RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &A,
                                  std::vector<LocalOrdinal>& dirichletRows, bool count_twos_as_dirichlet=false) {
      typedef typename Teuchos::ScalarTraits<Scalar>::magnitudeType MT;
      dirichletRows.resize(0);
      for(size_t i=0; i<A->getLocalNumRows(); i++) {
        Teuchos::ArrayView<const LocalOrdinal> indices;
        Teuchos::ArrayView<const Scalar> values;
        A->getLocalRowView(i,indices,values);
        int nnz=0;
        for (size_t j=0; j<(size_t)indices.size(); j++) {
          if (Teuchos::ScalarTraits<Scalar>::magnitude(values[j]) > Teuchos::ScalarTraits<MT>::eps()) {
            nnz++;
          }
        }
        if (nnz == 1 || (count_twos_as_dirichlet && nnz == 2)) {
          dirichletRows.push_back(i);
        }
      }
    }

    // Applies Ones-and-Zeros to matrix rows
    // Takes a vector of row indices
    static void ApplyOAZToMatrixRows(Teuchos::RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& A,
                               const std::vector<LocalOrdinal>& dirichletRows) {
      RCP<const Map> Rmap = A->getRowMap();
      RCP<const Map> Cmap = A->getColMap();
      Scalar one  =Teuchos::ScalarTraits<Scalar>::one();
      Scalar zero =Teuchos::ScalarTraits<Scalar>::zero();

      for(size_t i=0; i<dirichletRows.size(); i++) {
        GlobalOrdinal row_gid = Rmap->getGlobalElement(dirichletRows[i]);

        Teuchos::ArrayView<const LocalOrdinal> indices;
        Teuchos::ArrayView<const Scalar> values;
        A->getLocalRowView(dirichletRows[i],indices,values);
        // NOTE: This won't work with fancy node types.
        Scalar* valuesNC = const_cast<Scalar*>(values.getRawPtr());
        for(size_t j=0; j<(size_t)indices.size(); j++) {
          if(Cmap->getGlobalElement(indices[j])==row_gid)
            valuesNC[j]=one;
          else
            valuesNC[j]=zero;
        }
      }
    }

    // Applies Ones-and-Zeros to matrix rows
    // Takes a Boolean array.
    static void ApplyOAZToMatrixRows(Teuchos::RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& A,
                                     const Teuchos::ArrayRCP<const bool>& dirichletRows) {
      TEUCHOS_ASSERT(A->isFillComplete());
      RCP<const Map> domMap = A->getDomainMap();
      RCP<const Map> ranMap = A->getRangeMap();
      RCP<const Map> Rmap = A->getRowMap();
      RCP<const Map> Cmap = A->getColMap();
      TEUCHOS_ASSERT(static_cast<size_t>(dirichletRows.size()) == Rmap->getLocalNumElements());
      const Scalar one  = Teuchos::ScalarTraits<Scalar>::one();
      const Scalar zero = Teuchos::ScalarTraits<Scalar>::zero();
      A->resumeFill();
      for(size_t i=0; i<(size_t) dirichletRows.size(); i++) {
        if (dirichletRows[i]){
          GlobalOrdinal row_gid = Rmap->getGlobalElement(i);

          Teuchos::ArrayView<const LocalOrdinal> indices;
          Teuchos::ArrayView<const Scalar> values;
          A->getLocalRowView(i,indices,values);

          Teuchos::ArrayRCP<Scalar> valuesNC(values.size());
          for(size_t j=0; j<(size_t)indices.size(); j++) {
            if(Cmap->getGlobalElement(indices[j])==row_gid)
              valuesNC[j]=one;
            else
              valuesNC[j]=zero;
          }
          A->replaceLocalValues(i,indices,valuesNC());
        }
      }
      A->fillComplete(domMap, ranMap);
    }

    // Zeros out rows
    // Takes a vector containg Dirichlet row indices
    static void ZeroDirichletRows(Teuchos::RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& A,
                                  const std::vector<LocalOrdinal>& dirichletRows,
                                  Scalar replaceWith=Teuchos::ScalarTraits<Scalar>::zero()) {
      for(size_t i=0; i<dirichletRows.size(); i++) {
        Teuchos::ArrayView<const LocalOrdinal> indices;
        Teuchos::ArrayView<const Scalar> values;
        A->getLocalRowView(dirichletRows[i],indices,values);
        // NOTE: This won't work with fancy node types.
        Scalar* valuesNC = const_cast<Scalar*>(values.getRawPtr());
        for(size_t j=0; j<(size_t)indices.size(); j++)
            valuesNC[j]=replaceWith;
      }
    }

    // Zeros out rows
    // Takes a Boolean ArrayRCP
    static void ZeroDirichletRows(Teuchos::RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& A,
                                  const Teuchos::ArrayRCP<const bool>& dirichletRows,
                                  Scalar replaceWith=Teuchos::ScalarTraits<Scalar>::zero()) {
      TEUCHOS_ASSERT(static_cast<size_t>(dirichletRows.size()) == A->getRowMap()->getLocalNumElements());
      for(size_t i=0; i<(size_t) dirichletRows.size(); i++) {
        if (dirichletRows[i]) {
          Teuchos::ArrayView<const LocalOrdinal> indices;
          Teuchos::ArrayView<const Scalar> values;
          A->getLocalRowView(i,indices,values);
          // NOTE: This won't work with fancy node types.
          Scalar* valuesNC = const_cast<Scalar*>(values.getRawPtr());
          for(size_t j=0; j<(size_t)indices.size(); j++)
            valuesNC[j]=replaceWith;
        }
      }
    }

    // Zeros out rows
    // Takes a Boolean ArrayRCP
    static void ZeroDirichletRows(Teuchos::RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& X,
                                  const Teuchos::ArrayRCP<const bool>& dirichletRows,
                                  Scalar replaceWith=Teuchos::ScalarTraits<Scalar>::zero()) {
      TEUCHOS_ASSERT(static_cast<size_t>(dirichletRows.size()) == X->getMap()->getLocalNumElements());
      for(size_t i=0; i<(size_t) dirichletRows.size(); i++) {
        if (dirichletRows[i]) {
          for(size_t j=0; j<X->getNumVectors(); j++)
            X->replaceLocalValue(i,j,replaceWith);
        }
      }
    }

    // Zeros out columns
    // Takes a Boolean vector
    static void ZeroDirichletCols(Teuchos::RCP<Matrix>& A,
                                  const Teuchos::ArrayRCP<const bool>& dirichletCols,
                                  Scalar replaceWith=Teuchos::ScalarTraits<Scalar>::zero()) {
      TEUCHOS_ASSERT(static_cast<size_t>(dirichletCols.size()) == A->getColMap()->getLocalNumElements());
      for(size_t i=0; i<A->getLocalNumRows(); i++) {
        Teuchos::ArrayView<const LocalOrdinal> indices;
        Teuchos::ArrayView<const Scalar> values;
        A->getLocalRowView(i,indices,values);
        // NOTE: This won't work with fancy node types.
        Scalar* valuesNC = const_cast<Scalar*>(values.getRawPtr());
        for(size_t j=0; j<static_cast<size_t>(indices.size()); j++)
          if (dirichletCols[indices[j]])
            valuesNC[j] = replaceWith;
      }
    }

    // Finds the OAZ Dirichlet rows for this matrix
    static void FindDirichletRowsAndPropagateToCols(Teuchos::RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > &A,
                                                    Teuchos::RCP<Xpetra::Vector<int,LocalOrdinal,GlobalOrdinal,Node> >& isDirichletRow,
                                                    Teuchos::RCP<Xpetra::Vector<int,LocalOrdinal,GlobalOrdinal,Node> >& isDirichletCol) {

      // Make sure A's RowMap == DomainMap
      if(!A->getRowMap()->isSameAs(*A->getDomainMap())) {
        throw std::runtime_error("UtilitiesBase::FindDirichletRowsAndPropagateToCols row and domain maps must match.");
      }
      RCP<const Xpetra::Import<LocalOrdinal,GlobalOrdinal,Node> > importer = A->getCrsGraph()->getImporter();
      bool has_import = !importer.is_null();

      // Find the Dirichlet rows
      std::vector<LocalOrdinal> dirichletRows;
      FindDirichletRows(A,dirichletRows);

#if 0
    printf("[%d] DirichletRow Ids = ",A->getRowMap()->getComm()->getRank());
      for(size_t i=0; i<(size_t) dirichletRows.size(); i++)
        printf("%d ",dirichletRows[i]);
    printf("\n");
    fflush(stdout);
#endif
      // Allocate all as non-Dirichlet
      isDirichletRow = Xpetra::VectorFactory<int,LocalOrdinal,GlobalOrdinal,Node>::Build(A->getRowMap(),true);
      isDirichletCol = Xpetra::VectorFactory<int,LocalOrdinal,GlobalOrdinal,Node>::Build(A->getColMap(),true);

      {
        Teuchos::ArrayRCP<int> dr_rcp = isDirichletRow->getDataNonConst(0);
        Teuchos::ArrayView<int> dr    = dr_rcp();
        Teuchos::ArrayRCP<int> dc_rcp = isDirichletCol->getDataNonConst(0);
        Teuchos::ArrayView<int> dc    = dc_rcp();
        for(size_t i=0; i<(size_t) dirichletRows.size(); i++) {
          dr[dirichletRows[i]] = 1;
          if(!has_import) dc[dirichletRows[i]] = 1;
        }
      }

      if(has_import)
        isDirichletCol->doImport(*isDirichletRow,*importer,Xpetra::CombineMode::ADD);

    }

    // This routine takes a BlockedMap and an Importer (assuming that the BlockedMap matches the source of the importer) and generates a BlockedMap corresponding
    // to the Importer's target map.  We assume that the targetMap is unique (which, is not a strict requirement of an Importer, but is here and no, we don't check)
    // This is largely intended to be used in repartitioning of blocked matrices
    static RCP<const Xpetra::BlockedMap<LocalOrdinal,GlobalOrdinal,Node> > GeneratedBlockedTargetMap(const Xpetra::BlockedMap<LocalOrdinal,GlobalOrdinal,Node> & sourceBlockedMap,
												     const Xpetra::Import<LocalOrdinal,GlobalOrdinal,Node> & Importer) {
      typedef Xpetra::Vector<int,LocalOrdinal,GlobalOrdinal,Node> IntVector;
      Xpetra::UnderlyingLib lib = sourceBlockedMap.lib();

      // De-stride the map if we have to (might regret this later)
      RCP<const Map> fullMap    = sourceBlockedMap.getMap();
      RCP<const Map> stridedMap = Teuchos::rcp_dynamic_cast<const Xpetra::StridedMap<LocalOrdinal,GlobalOrdinal,Node> >(fullMap);
      if(!stridedMap.is_null()) fullMap = stridedMap->getMap();

      // Initial sanity checking for map compatibil
      const size_t numSubMaps = sourceBlockedMap.getNumMaps();
      if(!Importer.getSourceMap()->isCompatible(*fullMap))
	throw std::runtime_error("GenerateBlockedTargetMap(): Map compatibility error");

      // Build an indicator vector
      RCP<IntVector> block_ids = Xpetra::VectorFactory<int,LocalOrdinal,GlobalOrdinal,Node>::Build(fullMap);

      for(size_t i=0; i<numSubMaps; i++) {
	RCP<const Map> map = sourceBlockedMap.getMap(i);

	for(size_t j=0; j<map->getLocalNumElements(); j++)  {
	  LocalOrdinal jj = fullMap->getLocalElement(map->getGlobalElement(j));
	  block_ids->replaceLocalValue(jj,(int)i);
	}
      }

      // Get the block ids for the new map
      RCP<const Map> targetMap = Importer.getTargetMap();
      RCP<IntVector> new_block_ids = Xpetra::VectorFactory<int,LocalOrdinal,GlobalOrdinal,Node>::Build(targetMap);
      new_block_ids->doImport(*block_ids,Importer,Xpetra::CombineMode::ADD);
      Teuchos::ArrayRCP<const int> dataRCP = new_block_ids->getData(0);
      Teuchos::ArrayView<const int> data = dataRCP();


      // Get the GIDs for each subblock
      Teuchos::Array<Teuchos::Array<GlobalOrdinal> > elementsInSubMap(numSubMaps);
      for(size_t i=0; i<targetMap->getLocalNumElements(); i++) {
	elementsInSubMap[data[i]].push_back(targetMap->getGlobalElement(i));
      }

      // Generate the new submaps
      std::vector<RCP<const Map> > subMaps(numSubMaps);
      for(size_t i=0; i<numSubMaps; i++) {
	subMaps[i] = Xpetra::MapFactory<LocalOrdinal,GlobalOrdinal,Node>::Build(lib,Teuchos::OrdinalTraits<GlobalOrdinal>::invalid(),elementsInSubMap[i](),targetMap->getIndexBase(),targetMap->getComm());
      }

      // Build the BlockedMap
      return rcp(new BlockedMap(targetMap,subMaps));

    }

    // Checks to see if the first chunk of the colMap is also the row map.  This simiplifies a bunch of
    // operation in coarsening
    static bool MapsAreNested(const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node>& rowMap, const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node>& colMap) {
      ArrayView<const GlobalOrdinal> rowElements = rowMap.getLocalElementList();
      ArrayView<const GlobalOrdinal> colElements = colMap.getLocalElementList();
      
      const size_t numElements = rowElements.size();
      
      if (size_t(colElements.size()) < numElements)
	return false;

      bool goodMap = true;
      for (size_t i = 0; i < numElements; i++)
	if (rowElements[i] != colElements[i]) {
	  goodMap = false;
	  break;
      }
      
      return goodMap;
    }




  }; // class Utils


  ///////////////////////////////////////////

} //namespace MueLu

#define MUELU_UTILITIESBASE_SHORT
#endif // MUELU_UTILITIESBASE_DECL_HPP
