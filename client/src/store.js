import { configureStore } from "@reduxjs/toolkit";
import EvaluationSlice from "./Slices/EvaluationSlice";
import StructureAnalysisSlice from "./Slices/StructureAnalysisSlice";

export const store = configureStore({
  reducer: {
    structure: StructureAnalysisSlice,
    evaluation: EvaluationSlice,
  },
});
