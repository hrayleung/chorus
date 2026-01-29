// Together's SDK optionally imports "parquetjs" for parquet file support.
// We don't need parquet support in Chorus, but Vite needs the module to exist at bundle time.
// Throwing here keeps runtime behavior consistent with "parquetjs" being absent.
throw new Error("parquetjs is not installed");

export {};
