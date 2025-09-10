import { Config } from '@stencil/core';

export const config: Config = {
  namespace: 'math-expression-drawer',
  outputTargets: [
    {
      type: 'dist',
      esmLoaderPath: '../loader',
    },
    {
      type: 'dist-custom-elements',
    },
    {
      type: 'docs-readme',
    },
    {
      type: 'www',
      serviceWorker: null,
    },
  ],
  devServer: {
    port: 3333,
    openBrowser: false,
  },
};