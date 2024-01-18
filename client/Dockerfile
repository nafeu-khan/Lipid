FROM node:18-alpine3.18

RUN apk update && apk add git

WORKDIR /app
COPY package.json .
RUN npm install
COPY . .

EXPOSE 5173
CMD [ "npm", "run", "dev" ]