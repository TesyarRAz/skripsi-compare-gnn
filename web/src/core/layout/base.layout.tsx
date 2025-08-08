import { Outlet } from 'react-router'

const BaseLayout = () => {
  return (
    <div className='min-h-screen h-screen max-h-screen w-screen'>
      <main className="size-full">
        <Outlet />
      </main>
    </div>
  )
}

export default BaseLayout