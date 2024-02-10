import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";
import { graph_data } from "../utility";

const graphData = JSON.parse(graph_data);
const URL = "http://127.0.0.1:8000";

export const evaluateModel = createAsyncThunk(
  "lipid/evaluateModel",
  async () => {
    try {
      const res = await fetch(`${URL}/eval/`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });
      const data = await res.json();

      return data;
      // return { data, mol_name };
    } catch (error) {
      console.log(error);
    }
  }
);

const initialState = {
  loading: false,
  data: undefined,
};

export const evaluation = createSlice({
  name: "eval",
  initialState,
  reducers: {},
  extraReducers: (builder) => {
    builder.addCase(evaluateModel.fulfilled, (state, { payload }) => {
      state.loading = false;
      state.data = payload;
    });
    builder.addCase(evaluateModel.rejected, (state) => {
      state.loading = false;
    });
    builder.addCase(evaluateModel.pending, (state) => {
      state.loading = true;
    });
  },
});

// Action creators are generated for each case reducer function
// export const { changeActiveLipid, changeNumOfComp, changeShowTable } =
//   evaluation.actions;

export default evaluation.reducer;
