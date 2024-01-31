import { configureStore } from "@reduxjs/toolkit";
import StructureAnalysisSlice from "./Slices/StructureAnalysisSlice";

export const store = configureStore({
  reducer: {
    structure: StructureAnalysisSlice,
  },
});
